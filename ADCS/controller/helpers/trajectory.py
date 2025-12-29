__all__ = ["Trajectory"]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional

class Trajectory:
    def __init__(self, t: np.ndarray, x: np.ndarray, u: np.ndarray, K: np.ndarray, S: np.ndarray) -> None:
        self.times = t
        self.states = x  # Shape: (nx, N)
        self.controls = u  # Shape: (nu, N-1) or (nu, N)
        self.gains = K
        self.costs = S

        self.start_time = t[0]
        self.end_time = t[-1]
        self.n_steps = len(t)

        self.state_dim = x.shape[0]
        self.ctrl_dim = u.shape[0]

    def is_valid_time(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time
    
    def get_state_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        dt = self.times[idx+1] - self.times[idx]
        if dt == 0: return self.states[:, idx]
        alpha = (t - self.times[idx]) / dt
        
        x0 = self.states[:, idx]
        x1 = self.states[:, idx+1]

        # Linear interpolation for most states
        state_interp = (1 - alpha) * x0 + alpha * x1
        
        # Normalized Lerp for Quaternion (indices 3:7)
        if self.state_dim >= 7:
            q0 = x0[3:7]
            q1 = x1[3:7]
            q_interp = (1 - alpha) * q0 + alpha * q1
            state_interp[3:7] = q_interp / np.linalg.norm(q_interp)
            
        return state_interp
    
    def get_control_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        # Handle case where controls might be 1 shorter than states (N-1 vs N)
        if idx >= self.controls.shape[1] - 1:
            return self.controls[:, -1]
            
        dt = self.times[idx+1] - self.times[idx]
        if dt == 0: return self.controls[:, idx]
        alpha = (t - self.times[idx]) / dt
        return (1 - alpha) * self.controls[:, idx] + alpha * self.controls[:, idx+1]
    
    def get_gain_at(self, t: float) -> np.ndarray:
        # Nearest neighbor interpolation for gains is standard for discrete LQR
        idx = (np.abs(self.times - t)).argmin()
        
        # Handle dimensions: K might be (N, nu, nx) or flattened/transposed
        if self.gains.ndim == 3:
            # Assuming shape (N, nu, nx) based on previous code context
            # We need to check if the time dimension is axis 0 or 2.
            # Planner returned (N-1, nu, nx). If we padded, might be (N, nu, nx).
            if self.gains.shape[0] == len(self.times) or self.gains.shape[0] == len(self.times)-1:
                 # Time is first axis
                 safe_idx = min(idx, self.gains.shape[0]-1)
                 return self.gains[safe_idx, :, :]
            else:
                 # Time is last axis
                 safe_idx = min(idx, self.gains.shape[2]-1)
                 return self.gains[:, :, safe_idx]
        else:
            # Fallback for flattened gains
            k_flat = self.gains[:, idx]
            return k_flat.reshape(self.ctrl_dim, self.state_dim)
    
    def compute_tracking_control(self, t: float, x_current: np.ndarray) -> np.ndarray:
        if not self.is_valid_time(t):
            raise ValueError(f"Time {t} is outside trajectory bounds [{self.start_time}, {self.end_time}]")
        
        x_ref = self.get_state_at(t)
        u_ref = self.get_control_at(t)
        K = self.get_gain_at(t)
        
        dx = x_current - x_ref
        
        # Quaternion error handling (shortest path)
        if self.state_dim >= 7:
            # If quaternions are antiparallel (dot < 0), flip sign of reference
            if np.dot(x_current[3:7], x_ref[3:7]) < 0:
                dx[3:7] = x_current[3:7] + x_ref[3:7] 

        return u_ref - K @ dx

    def get_state_input_gain(self, t: float):
        """Helper to get state, input, and gain efficiently."""
        return self.get_state_at(t), self.get_control_at(t), self.get_gain_at(t), None

    def get_plotting_data(self) -> Dict[str, np.ndarray]:
        return {
            "time": self.times,
            "state": self.states,
            "control": self.controls,
            "cost": self.costs
        }
    
    def _get_idx(self, t: float) -> int:
        if t >= self.end_time: 
            return self.n_steps - 2
        idx = np.searchsorted(self.times, t, side='right') - 1
        return max(0, min(idx, self.n_steps - 2))

    # --- Visualization Methods ---

    def plot_eci_trajectory(self, 
                            body_axis: np.ndarray = np.array([0, 0, 1]), 
                            stride: int = 1,
                            show: bool = True):
        """
        Visualizes the trajectory of a specific body-frame vector (e.g., boresight)
        as it rotates in the ECI frame over time.
        
        Args:
            body_axis: Vector in body frame to trace (default: Z-axis/Boresight).
            stride: Plot every n-th point to reduce clutter for dense trajectories.
            show: Whether to call plt.show() at the end.
        """
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract quaternions. Assumes state layout: [w(3), q(4), h(3)]
        # Quaternions at indices 3:7 [scalar, x, y, z]
        
        # Handle state shape (nx, N) vs (N, nx)
        if self.states.shape[0] == self.state_dim:
            # (nx, N) - Column Major
            quats = self.states[3:7, ::stride]
            times = self.times[::stride]
        else:
            # (N, nx) - Row Major
            quats = self.states[::stride, 3:7].T
            times = self.times[::stride]
        
        # Normalize body axis
        v_body = body_axis / np.linalg.norm(body_axis)
        
        # Rotate body vector to ECI for all steps
        v_eci_list = []
        for i in range(quats.shape[1]):
            q = quats[:, i]
            # normalize quaternion just in case
            q = q / np.linalg.norm(q)
            v_eci = self._rotate_vector(q, v_body)
            v_eci_list.append(v_eci)
            
        v_eci = np.array(v_eci_list).T # Shape (3, N_plotted)
        
        # 1. Plot Unit Sphere for reference
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = np.cos(u)*np.sin(v)
        y_sphere = np.sin(u)*np.sin(v)
        z_sphere = np.cos(v)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.15)
        
        # 2. Plot Trajectory Trace
        # Color points by time to visualize direction and speed
        p = ax.scatter(v_eci[0, :], v_eci[1, :], v_eci[2, :], c=times, cmap='viridis', s=10, label='Trace')
        
        # 3. Mark Start and End
        ax.scatter(v_eci[0, 0], v_eci[1, 0], v_eci[2, 0], color='green', s=100, marker='o', label='Start')
        ax.scatter(v_eci[0, -1], v_eci[1, -1], v_eci[2, -1], color='red', s=100, marker='X', label='End')
        
        # Labels and Style
        ax.set_xlabel("ECI X")
        ax.set_ylabel("ECI Y")
        ax.set_zlabel("ECI Z")
        ax.set_title(f"Trajectory of Body Axis {body_axis} in ECI")
        ax.legend()
        
        # Force equal aspect ratio so sphere looks like a sphere
        self._set_axes_equal(ax)
        
        cbar = fig.colorbar(p, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Time (J2000)')
        
        if show:
            plt.show()

    def _rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotates vector v by quaternion q (active rotation, body->ECI).
        Formula: v_rotated = v + 2*cross(q_vec, cross(q_vec, v) + q_scalar*v)
        Assuming q = [w, x, y, z] (scalar first).
        """
        q_scalar = q[0]
        q_vec = q[1:]
        
        # Rodrigues rotation formula
        t = 2 * np.cross(q_vec, v)
        v_prime = v + q_scalar * t + np.cross(q_vec, t)
        
        return v_prime

    def _set_axes_equal(self, ax):
        """Helper to set 3D axes aspect ratio to equal."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a cube with side length = max_range
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
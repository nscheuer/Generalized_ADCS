"""
Trajectory representation for ALTRO planner output.

This module provides the Trajectory class for storing, interpolating, and
visualizing planned trajectories from the ALTRO optimizer.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, Tuple, Callable
from numpy.typing import NDArray

from ADCS.helpers.math_helpers import quat_diff, quat_to_vec3


class Trajectory:
    """
    Container for trajectory optimization results with interpolation support.

    Stores time series of states, controls, feedback gains, and cost-to-go values
    from the ALTRO planner. Provides interpolation methods for use in tracking control.

    Attributes:
        times: Time stamps for trajectory points (J2000 centuries)
        states: State trajectory, shape (state_dim, n_steps) or (n_steps, state_dim)
        controls: Control trajectory, shape (ctrl_dim, n_steps-1) or (n_steps-1, ctrl_dim)
        gains: Feedback gain matrices K for TVLQR tracking
        costs: Cost-to-go values at each timestep
        start_time: First time in trajectory
        end_time: Last time in trajectory
        n_steps: Number of time points
        state_dim: Dimension of state vector
        ctrl_dim: Dimension of control vector
    """

    # Class-level type annotations
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    gains: NDArray[np.float64]
    costs: NDArray[np.float64]
    start_time: float
    end_time: float
    n_steps: int
    state_dim: int
    ctrl_dim: int
    _is_row_major: bool

    def __init__(
        self,
        t: NDArray[np.float64],
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        K: NDArray[np.float64],
        S: NDArray[np.float64]
    ) -> None:
        """
        Initialize trajectory from planner output.

        Args:
            t: Time array of shape (n_steps,)
            x: State array, either (n_steps, state_dim) or (state_dim, n_steps)
            u: Control array, either (n_steps-1, ctrl_dim) or (ctrl_dim, n_steps-1)
            K: Feedback gains array
            S: Cost-to-go array
        """
        self.times = t
        self.states = x
        self.controls = u
        self.gains = K
        self.costs = S

        self.start_time = float(t[0])
        self.end_time = float(t[-1])
        self.n_steps = len(t)

        # Robust Dimension Detection
        # Check if Axis 0 matches time steps (Row-Major: N x nx)
        if x.shape[0] == self.n_steps:
            self.state_dim = x.shape[1]
            self._is_row_major = True
        else:
            self.state_dim = x.shape[0]
            self._is_row_major = False

        # Same check for controls
        if u.shape[0] == self.n_steps or u.shape[0] == self.n_steps - 1:
            self.ctrl_dim = u.shape[1]
        else:
            self.ctrl_dim = u.shape[0]

    def is_valid_time(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time
    
    def get_state_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        dt = self.times[idx+1] - self.times[idx]
        
        # Get raw states based on layout
        if self._is_row_major:
            x0 = self.states[idx, :]
            x1 = self.states[idx+1, :]
        else:
            x0 = self.states[:, idx]
            x1 = self.states[:, idx+1]

        if dt == 0: return x0
        alpha = (t - self.times[idx]) / dt

        # Linear Interpolation
        state_interp = (1 - alpha) * x0 + alpha * x1
        
        # Normalize Quaternion (indices 3:7) if state is large enough
        if self.state_dim >= 7:
            # Handle standard ADCS state vector: [w(3), q(4), h(3)]
            q0 = x0[3:7]
            q1 = x1[3:7]
            # Simple lerp then normalize is sufficient for small steps
            q_interp = (1 - alpha) * q0 + alpha * q1
            if np.linalg.norm(q_interp) > 1e-9:
                state_interp[3:7] = q_interp / np.linalg.norm(q_interp)
            
        return state_interp
    
    def get_control_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        
        # Helper to extract u at index i handling layout
        def get_u(i):
            # Clamp index for N-1 controls
            limit = self.controls.shape[0] if self._is_row_major else self.controls.shape[1]
            if i >= limit: i = limit - 1
            
            if self._is_row_major:
                return self.controls[i, :]
            return self.controls[:, i]

        u0 = get_u(idx)
        u1 = get_u(idx+1)
            
        dt = self.times[idx+1] - self.times[idx]
        if dt == 0: return u0
        alpha = (t - self.times[idx]) / dt
        return (1 - alpha) * u0 + alpha * u1
    
    def get_gain_at(self, t: float) -> np.ndarray:
        idx = (np.abs(self.times - t)).argmin()
        
        # Handle Gain shape conventions
        # Expected: (N, nu, nx) [Row Major Time] OR (nu, nx, N) [Col Major Time]
        
        # Check dimensions
        if self.gains.ndim == 3:
            if self.gains.shape[0] >= self.n_steps - 1:
                # Time is first axis
                safe_idx = min(idx, self.gains.shape[0]-1)
                return self.gains[safe_idx, :, :]
            elif self.gains.shape[2] >= self.n_steps - 1:
                # Time is last axis
                safe_idx = min(idx, self.gains.shape[2]-1)
                return self.gains[:, :, safe_idx]
        
        # Fallback for flattened
        k_flat = self.gains[:, idx]
        return k_flat.reshape(self.ctrl_dim, self.state_dim-1)
    
    def compute_tracking_control(self, t: float, x_current: np.ndarray) -> np.ndarray:
        if not self.is_valid_time(t):
            raise ValueError(f"Time {t} is outside bounds")
        
        x_ref = self.get_state_at(t)
        u_ref = self.get_control_at(t)
        K = self.get_gain_at(t)
        
        def state_diff(x_curr: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
            # Initialize 9-element error state
            dx = np.zeros(9) 
            
            # 1. Angular Velocity (Indices 0, 1, 2)
            # Use 0:3 to include index 2
            dx[0:3] = x_curr[0:3] - x_ref[0:3]
            
            # 2. Attitude Error (Indices 3, 4, 5)
            # Ensure quat_diff returns q_ref^(-1) * q_curr
            q_err = quat_diff(x_curr[3:7], x_ref[3:7])
            
            # CRITICAL: LQR usually assumes linear error d_theta = 2 * vector_part(q_err)
            # If quat_to_vec3 just returns (x,y,z), you might need to multiply by 2.
            # If your K was generated assuming d_q ~ [1, d_theta/2], keep the factor of 2.
            dx[3:6] = 2 * quat_to_vec3(q_err) 
            
            # 3. Wheel Momentum (Indices 6, 7, 8 in error state; 7, 8, 9 in full state)
            # Use 6:9 (dest) and 7:10 (source) to include the last element
            dx[6:9] = x_curr[7:10] - x_ref[7:10]
            
            return dx

        dx = state_diff(x_current, x_ref)
        
        # Apply Control Law
        return u_ref - K @ dx

    def get_state_input_gain(
        self, t: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], None]:
        """Get state, control, and gain at time t for tracking control."""
        return self.get_state_at(t), self.get_control_at(t), self.get_gain_at(t), None

    def get_plotting_data(self) -> Dict[str, NDArray[np.float64]]:
        """Return dictionary of trajectory data for plotting."""
        return {
            "time": self.times,
            "state": self.states,
            "control": self.controls,
            "cost": self.costs
        }

    def _get_idx(self, t: float) -> int:
        """Find the trajectory index for interpolation at time t."""
        if t >= self.end_time:
            return self.n_steps - 2
        idx = np.searchsorted(self.times, t, side='right') - 1
        return max(0, min(idx, self.n_steps - 2))

    # --- Visualization Methods ---

    def plot_eci_trajectory(self, 
                            body_axis: np.ndarray = np.array([0, 0, 1]), 
                            stride: int = 1,
                            show: bool = True):
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- Robust State Extraction ---
        # We need quaternions (indices 3:7) for every time step.
        # Goal: quats shape (4, N_points)
        
        # Check if Axis 0 is Time (N, nx)
        if self.states.shape[0] == self.n_steps:
            # Row-Major: Slice rows by stride, grab cols 3:7, Transpose to (4, N)
            quats = self.states[::stride, 3:7].T
        # Check if Axis 1 is Time (nx, N)
        elif self.states.shape[1] == self.n_steps:
            # Col-Major: Grab cols 3:7, slice cols by stride -> (4, N)
            quats = self.states[3:7, ::stride]
        else:
            raise ValueError(f"State shape {self.states.shape} does not match n_steps={self.n_steps}")

        times = self.times[::stride]
        
        # Ensure lengths match exactly (handle any potential off-by-one from slicing)
        # Usually standard slicing [::s] is consistent for both arrays of same len
        
        v_body = body_axis / np.linalg.norm(body_axis)
        
        # Rotate body vector to ECI
        v_eci_list = []
        for i in range(quats.shape[1]):
            q = quats[:, i]
            # Safety normalize
            norm = np.linalg.norm(q)
            if norm > 1e-6: q = q / norm
            
            v_eci = self._rotate_vector(q, v_body)
            v_eci_list.append(v_eci)
            
        v_eci = np.array(v_eci_list).T # (3, N_plotted)
        
        # --- DEBUG PRINT (Optional) ---
        # print(f"DEBUG: Plotting {v_eci.shape[1]} points. Time len: {len(times)}")

        # Plot Sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = np.cos(u)*np.sin(v)
        y_sphere = np.sin(u)*np.sin(v)
        z_sphere = np.cos(v)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.15)
        
        # Plot Trace
        # c argument needs to match x/y size. 
        p = ax.scatter(v_eci[0, :], v_eci[1, :], v_eci[2, :], c=times, cmap='viridis', s=10, label='Trace')
        
        # Markers
        ax.scatter(v_eci[0, 0], v_eci[1, 0], v_eci[2, 0], color='green', s=100, marker='o', label='Start')
        ax.scatter(v_eci[0, -1], v_eci[1, -1], v_eci[2, -1], color='red', s=100, marker='X', label='End')
        
        ax.set_xlabel("ECI X")
        ax.set_ylabel("ECI Y")
        ax.set_zlabel("ECI Z")
        ax.set_title(f"Trajectory of Body Axis {body_axis} in ECI")
        ax.legend()
        
        self._set_axes_equal(ax)
        cbar = fig.colorbar(p, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Time (J2000)')
        
        if show:
            plt.show()

    def _rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_scalar = q[0]
        q_vec = q[1:]
        t = 2 * np.cross(q_vec, v)
        v_prime = v + q_scalar * t + np.cross(q_vec, t)
        return v_prime

    def _set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
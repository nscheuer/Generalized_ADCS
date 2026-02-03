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
from ADCS.helpers.simresults import SimulationResults, RunResults
from ADCS.satellite_hardware.satellite import Satellite

class Trajectory:
    r"""
    Container for trajectory optimization results produced by the ALTRO planner.

    This class stores discrete-time trajectories of states, controls, feedback
    gains, and cost-to-go values, and provides interpolation utilities for
    real-time tracking control.

    The reference trajectory defines a nominal solution
    :math:`(\mathbf{x}^\ast(t), \mathbf{u}^\ast(t))` to a nonlinear optimal control
    problem. A time-varying LQR feedback law is applied as

    .. math::

        \mathbf{u}(t) =
        \mathbf{u}^\ast(t) - \mathbf{K}(t)\,\delta\mathbf{x}(t)

    where :math:`\delta\mathbf{x}` is a reduced-dimension error state that replaces
    the quaternion with a minimal 3D attitude error.

    The class supports both row-major and column-major storage layouts and
    automatically detects dimensions.

    :param t:
        Time stamps associated with the trajectory samples.

    :type t:
        numpy.ndarray

    :param x:
        State trajectory array, either time-major or state-major.

    :type x:
        numpy.ndarray

    :param u:
        Control trajectory array, either time-major or control-major.

    :type u:
        numpy.ndarray

    :param K:
        Feedback gain matrices for TVLQR tracking.

    :type K:
        numpy.ndarray

    :param S:
        Cost-to-go values along the trajectory.

    :type S:
        numpy.ndarray

    :param use_disturbance_estimation:
        Flag enabling augmented error state with disturbance estimation.

    :type use_disturbance_estimation:
        bool

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
        S: NDArray[np.float64],
        use_disturbance_estimation: bool = False
    ) -> None:
        """
        Initialize trajectory from planner output.

        Args:
            t: Time array of shape (n_steps,)
            x: State array, either (n_steps, state_dim) or (state_dim, n_steps)
            u: Control array, either (n_steps-1, ctrl_dim) or (ctrl_dim, n_steps-1)
            K: Feedback gains array
            S: Cost-to-go array
            use_disturbance_estimation: If True, gains have 3 extra columns for
                disturbance estimation (KwDist mode, tracking_LQR_formulation=2)
        """
        self.times = t
        self.states = x
        self.controls = u
        self.gains = K
        self.costs = S
        self.use_disturbance_estimation = use_disturbance_estimation

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

        # Disturbance estimate for KwDist mode
        if use_disturbance_estimation:
            self._dist_estimate = np.zeros(3)

    def is_valid_time(self, t: float) -> bool:
        r"""
        Check whether a time lies within the trajectory bounds.

        :param t:
            Query time.

        :type t:
            float

        :return:
            True if the time is within the trajectory interval.

        :rtype:
            bool

        """
        return self.start_time <= t <= self.end_time
    
    def get_state_at(self, t: float) -> np.ndarray:
        r"""
        Interpolate the reference state at a given time.

        Linear interpolation is applied between neighboring samples. Quaternion
        components are renormalized to preserve unit norm.

        :param t:
            Query time.

        :type t:
            float

        :return:
            Interpolated reference state vector.

        :rtype:
            numpy.ndarray

        """
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
        r"""
        Interpolate the reference control at a given time.

        The control trajectory is interpolated linearly between neighboring
        time samples.

        :param t:
            Query time.

        :type t:
            float

        :return:
            Interpolated reference control vector.

        :rtype:
            numpy.ndarray

        """
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
        r"""
        Retrieve the feedback gain matrix at a given time.

        The gain matrix :math:`\mathbf{K}(t)` maps the reduced error state
        to control corrections. Multiple storage conventions are supported,
        including time-major, control-major, and flattened formats.

        :param t:
            Query time.

        :type t:
            float

        :return:
            Feedback gain matrix corresponding to the query time.

        :rtype:
            numpy.ndarray

        """
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
        
        # Fallback for flattened gain storage.
        # Gain matrix K maps error state (reduced) to control:
        #   K is (n_ctrl, n_err) where n_err = state_dim - 1
        #   The -1 accounts for quaternion 4D → 3D reduction in error state.
        #
        # For KwDist mode (disturbance estimation), gains will be
        # (n_ctrl, n_err + 3) but this is handled by checking actual shape.
        k_flat = self.gains[:, idx]
        if self.use_disturbance_estimation:
            # KwDist gains: (ctrl_dim, state_dim - 1 + 3)
            error_dim_with_dist = self.state_dim - 1 + 3
            return k_flat.reshape(self.ctrl_dim, error_dim_with_dist)
        else:
            return k_flat.reshape(self.ctrl_dim, self.state_dim - 1)
    
    def compute_tracking_control(self, t: float, x_current: np.ndarray) -> np.ndarray:
        r"""
        Compute the tracking control input at a given time and state.

        The control law is

        .. math::

            \mathbf{u} =
            \mathbf{u}^\ast(t) - \mathbf{K}(t)\,\delta\mathbf{x}

        where :math:`\delta\mathbf{x}` is the reduced error state computed from
        the current and reference states.

        :param t:
            Query time.

        :type t:
            float

        :param x_current:
            Current full state vector.

        :type x_current:
            numpy.ndarray

        :return:
            Control input computed by the tracking controller.

        :rtype:
            numpy.ndarray

        """
        if not self.is_valid_time(t):
            raise ValueError(f"Time {t} is outside bounds")

        x_ref = self.get_state_at(t)
        u_ref = self.get_control_at(t)
        K = self.get_gain_at(t)

        dx = self._state_diff(x_current, x_ref)

        if self.use_disturbance_estimation:
            # KwDist mode: augment error state with disturbance estimate
            dx_aug = np.concatenate([dx, self._dist_estimate])
            return u_ref - K @ dx_aug
        else:
            return u_ref - K @ dx

    def _state_diff(self, x_curr: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        r"""
        Compute the reduced error state for TVLQR feedback.

        The error state is defined as

        .. math::

            \delta\mathbf{x} =
            \begin{bmatrix}
                \boldsymbol{\omega} - \boldsymbol{\omega}^\ast \\
                2\,\mathrm{vec}\!\left(q_\text{ref}^{-1} \otimes q\right) \\
                \mathbf{h} - \mathbf{h}^\ast
            \end{bmatrix}

        where the quaternion error is reduced from 4D to 3D.

        :param x_curr:
            Current state vector.

        :type x_curr:
            numpy.ndarray

        :param x_ref:
            Reference state vector.

        :type x_ref:
            numpy.ndarray

        :return:
            Reduced-dimension error state vector.

        :rtype:
            numpy.ndarray

        """
        # Number of reaction wheels: full state = 7 + n_rw
        n_rw = self.state_dim - 7
        error_dim = 6 + n_rw

        dx = np.zeros(error_dim)

        # 1. Angular Velocity Error (indices 0:3)
        dx[0:3] = x_curr[0:3] - x_ref[0:3]

        # 2. Attitude Error (indices 3:6)
        # quat_diff returns q_ref^(-1) * q_curr
        q_err = quat_diff(x_ref[3:7], x_curr[3:7])
        # LQR assumes linearized error: d_theta = 2 * vector_part(q_err)
        dx[3:6] = quat_to_vec3(q_err)

        # 3. RW Momentum Error (indices 6:6+n_rw, from full state 7:7+n_rw)
        if n_rw > 0:
            dx[6:6+n_rw] = x_curr[7:7+n_rw] - x_ref[7:7+n_rw]

        return dx

    def update_disturbance_estimate(self, dist_torque: np.ndarray) -> None:
        r"""
        Update the internal disturbance torque estimate.

        This method is used when disturbance estimation is enabled in the
        tracking controller.

        :param dist_torque:
            Estimated disturbance torque in the body frame.

        :type dist_torque:
            numpy.ndarray

        :return:
            None.

        :rtype:
            None

        """
        if self.use_disturbance_estimation:
            self._dist_estimate = np.asarray(dist_torque).flatten()[:3]

    def get_state_input_gain(
        self, t: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], None]:
        r"""
        Return reference state, control, and gain at a given time.

        This method provides a unified interface for tracking controllers.

        :param t:
            Query time.

        :type t:
            float

        :return:
            Tuple containing reference state, reference control, gain matrix,
            and a placeholder value.

        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, None]

        """
        return self.get_state_at(t), self.get_control_at(t), self.get_gain_at(t), None

    def get_plotting_data(self) -> Dict[str, NDArray[np.float64]]:
        r"""
        Return trajectory data packaged for plotting.

        :return:
            Dictionary containing time, state, control, and cost arrays.

        :rtype:
            dict

        """
        return {
            "time": self.times,
            "state": self.states,
            "control": self.controls,
            "cost": self.costs
        }

    def _get_idx(self, t: float) -> int:
        r"""
        Determine the lower index for interpolation at a given time.

        :param t:
            Query time.

        :type t:
            float

        :return:
            Index corresponding to the interval containing the query time.

        :rtype:
            int

        """
        if t >= self.end_time:
            return self.n_steps - 2
        idx = np.searchsorted(self.times, t, side='right') - 1
        return max(0, min(idx, self.n_steps - 2))

    # --- Visualization Methods ---

    def plot_eci_trajectory(self, 
                            body_axis: np.ndarray = np.array([0, 0, 1]), 
                            stride: int = 1,
                            show: bool = True):
        r"""
        Plot the trajectory of a body-fixed axis expressed in the ECI frame.

        The body axis is rotated into the inertial frame using the quaternion
        trajectory, producing a 3D curve on the unit sphere.

        :param body_axis:
            Body-frame axis to visualize.

        :type body_axis:
            numpy.ndarray

        :param stride:
            Subsampling stride for plotting.

        :type stride:
            int

        :param show:
            Flag indicating whether to display the plot immediately.

        :type show:
            bool

        :return:
            None.

        :rtype:
            None

        """
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

    
    def to_simulation_results(
        self,
        satellite: Satellite,
        target: Optional[np.ndarray] = None,
        w_target: Optional[np.ndarray] = None,
        *,
        include_cost: bool = False,
    ) -> SimulationResults:

        time_J2000 = np.asarray(self.times)
        time_s = (time_J2000 - time_J2000[0]) * 36525.0 * 86400.0
        N = len(time_J2000)

        if self._is_row_major:
            state_hist = np.asarray(self.states)
            control_hist = np.asarray(self.controls)
        else:
            state_hist = np.asarray(self.states.T)
            control_hist = np.asarray(self.controls.T)

        run = RunResults(
            satellite=satellite,
            time_J2000=time_J2000,
            time_s=time_s,
            state_hist=state_hist,
            control_hist=control_hist,
        )

        if target is not None:
            target = np.asarray(target)
            if target.ndim == 1:
                run.target_hist = [target.copy() for _ in range(N)]
            else:
                run.target_hist = [target[k].copy() for k in range(N)]

        if w_target is not None:
            w_target = np.asarray(w_target)
            if w_target.ndim == 1:
                run.w_target_hist = [w_target.copy() for _ in range(N)]
            else:
                run.w_target_hist = [w_target[k].copy() for k in range(N)]

        if include_cost and self.costs is not None:
            costs = np.asarray(self.costs)
            run.target_hist = [np.array([costs[k]]) for k in range(N)]

        return SimulationResults(runs=[run])

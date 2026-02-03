"""
Live Visualization for Python ALILQR Trajectory Optimization.

Provides real-time, non-blocking visualization of the optimization process
showing trajectory evolution across iterations.

Features:
- Quaternion evolution
- Angular velocity trajectory  
- Control inputs (any number of channels)
- Angle error to goal
- Iteration labels and convergence metrics
- Non-blocking updates (doesn't pause optimization)

Usage:
    from ADCS.controller.helpers.live_planner_viz import LivePlannerViz
    
    viz = LivePlannerViz(goal_vector_eci=np.array([0, 0, 1]))
    viz.start()
    
    py_alilqr = PythonALILQR(planner, debug_callback=viz.update)
    result = py_alilqr.optimize(...)
    
    viz.finish()  # Keep plot open for inspection
"""
from __future__ import annotations

__all__ = ["LivePlannerViz", "create_live_callback"]

import numpy as np
from typing import Optional, Tuple, List, Callable
from numpy.typing import NDArray
import warnings

# Handle matplotlib backend for non-blocking
import matplotlib
try:
    # Try to use a non-blocking backend
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def quat_to_rotation_matrix(q: NDArray) -> NDArray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R


def compute_angle_error(Xset: NDArray, goal_vec_eci: NDArray, body_vec: NDArray) -> NDArray:
    """
    Compute pointing angle error at each timestep.
    
    Parameters
    ----------
    Xset : (state_dim, N) array
        State trajectory with quaternion at indices 3:7
    goal_vec_eci : (3,) or (3, N) or (4,) or (4, N) array
        Goal vector in ECI frame (3D) or goal quaternion (4D)
    body_vec : (3,) array
        Body-frame vector to align with goal
        
    Returns
    -------
    angles : (N,) array
        Angle error in degrees at each timestep
    """
    N = Xset.shape[1]
    angles = np.zeros(N)
    
    goal_vec_eci = np.asarray(goal_vec_eci)
    body_vec = np.asarray(body_vec).flatten()
    
    # Check if goal is quaternion (4D) or vector (3D)
    if goal_vec_eci.ndim == 1:
        is_quat_goal = (len(goal_vec_eci) == 4)
        goal_is_time_varying = False
    else:
        is_quat_goal = (goal_vec_eci.shape[0] == 4)
        goal_is_time_varying = (goal_vec_eci.shape[1] == N)
    
    for k in range(N):
        q = Xset[3:7, k]
        q = q / np.linalg.norm(q)  # Normalize
        R = quat_to_rotation_matrix(q)
        
        if is_quat_goal:
            # Quaternion goal: compute angle between quaternions
            if goal_is_time_varying:
                q_goal = goal_vec_eci[:, k]
            else:
                # Handle both 1D (4,) and 2D (4, N) with mismatched N
                if goal_vec_eci.ndim == 2:
                    q_goal = goal_vec_eci[:, 0]  # Use first column
                else:
                    q_goal = goal_vec_eci
            q_goal = q_goal / np.linalg.norm(q_goal)
            
            # Angle between quaternions: 2 * arccos(|q1 · q2|)
            dot = abs(np.dot(q, q_goal))
            dot = min(dot, 1.0)  # Clamp for numerical stability
            angles[k] = np.degrees(2 * np.arccos(dot))
        else:
            # Vector goal: compute angle between body vector and goal in ECI
            if goal_is_time_varying:
                g = goal_vec_eci[:, k]
            else:
                # Handle both 1D (3,) and 2D (3, N) with mismatched N
                if goal_vec_eci.ndim == 2:
                    g = goal_vec_eci[:, 0]  # Use first column
                else:
                    g = goal_vec_eci[:3] if len(goal_vec_eci) > 3 else goal_vec_eci
            
            g = g / np.linalg.norm(g)
            
            # Transform body vector to ECI
            body_in_eci = R @ body_vec
            body_in_eci = body_in_eci / np.linalg.norm(body_in_eci)
            
            # Angle between vectors
            dot = np.clip(np.dot(body_in_eci, g), -1.0, 1.0)
            angles[k] = np.degrees(np.arccos(dot))
    
    return angles


def get_control_colors(n_controls: int) -> List[str]:
    """Generate distinct colors for control channels."""
    # Use a colormap that provides good distinction
    if n_controls <= 10:
        # Tab10 colormap for up to 10 controls
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(n_controls)]
    else:
        # Use a continuous colormap for more
        cmap = plt.cm.viridis
        return [cmap(i / (n_controls - 1)) for i in range(n_controls)]


def get_control_labels(n_controls: int, actuator_names: Optional[List[str]] = None) -> List[str]:
    """Generate labels for control channels."""
    if actuator_names is not None and len(actuator_names) == n_controls:
        return actuator_names
    return [f'u{i+1}' for i in range(n_controls)]


class LivePlannerViz:
    """
    Live visualization for trajectory optimization.
    
    Creates a non-blocking matplotlib figure that updates in real-time
    as optimization progresses.
    
    Parameters
    ----------
    goal_vector_eci : array-like, optional
        Goal pointing direction in ECI (3D) or goal quaternion (4D)
        If None, angle error panel shows quaternion magnitude instead
    body_vector : array-like, optional
        Body-frame vector to align (default: [0, 0, 1] i.e., +Z)
    dt : float, optional
        Time step for x-axis scaling (default: 1.0)
    update_interval : int, optional
        Update plot every N iterations (default: 1)
    figsize : tuple, optional
        Figure size (default: (14, 10))
    dark_mode : bool, optional
        Use dark background (default: False)
    n_controls : int, optional
        Number of control channels. If None, auto-detected on first update.
    actuator_names : list of str, optional
        Names for each actuator (for legend labels)
    """
    
    def __init__(
        self,
        goal_vector_eci: Optional[NDArray] = None,
        body_vector: Optional[NDArray] = None,
        dt: float = 1.0,
        update_interval: int = 1,
        figsize: Tuple[float, float] = (14, 10),
        dark_mode: bool = False,
        n_controls: Optional[int] = None,
        actuator_names: Optional[List[str]] = None,
        umax: Optional[NDArray] = None
    ):
        self.goal_vector_eci = goal_vector_eci
        self.body_vector = body_vector if body_vector is not None else np.array([0, 0, 1])
        self.dt = dt
        self.update_interval = update_interval
        self.figsize = figsize
        self.dark_mode = dark_mode
        self.n_controls = n_controls
        self.actuator_names = actuator_names
        self.umax = umax  # Control limits for scaling plot
        
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.texts = {}
        
        self.iteration_count = 0
        self.cost_history = []
        self.cmax_history = []
        self.grad_history = []
        
        self._is_started = False
        self._controls_initialized = False
    
    def set_goal_vectors(self, goal_vector_eci: NDArray) -> None:
        """
        Update the goal vectors for time-varying goals.
        
        Call this when switching between passes with different N.
        
        Parameters
        ----------
        goal_vector_eci : (3,) or (3, N) array
            New goal vector(s) in ECI frame
        """
        self.goal_vector_eci = goal_vector_eci
        
    def start(self) -> None:
        """Initialize and show the figure."""
        if self._is_started:
            return
            
        # Enable interactive mode
        plt.ion()
        
        # Create figure with gridspec
        if self.dark_mode:
            plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.25)
        
        # Create subplots
        self.axes['omega'] = self.fig.add_subplot(gs[0, 0])
        self.axes['quat'] = self.fig.add_subplot(gs[0, 1])
        self.axes['control'] = self.fig.add_subplot(gs[1, 0])
        self.axes['angle_error'] = self.fig.add_subplot(gs[1, 1])
        self.axes['convergence'] = self.fig.add_subplot(gs[2, :])
        
        # Initialize empty lines for each subplot
        colors = {'r': '#e74c3c', 'g': '#2ecc71', 'b': '#3498db', 'k': '#2c3e50', 'y': '#f39c12'}
        
        # Angular velocity
        ax = self.axes['omega']
        self.lines['omega_x'], = ax.plot([], [], color=colors['r'], label='ωx', linewidth=1.5)
        self.lines['omega_y'], = ax.plot([], [], color=colors['g'], label='ωy', linewidth=1.5)
        self.lines['omega_z'], = ax.plot([], [], color=colors['b'], label='ωz', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Quaternion
        ax = self.axes['quat']
        self.lines['q0'], = ax.plot([], [], color=colors['k'], label='q₀ (w)', linewidth=1.5)
        self.lines['q1'], = ax.plot([], [], color=colors['r'], label='q₁ (x)', linewidth=1.5)
        self.lines['q2'], = ax.plot([], [], color=colors['g'], label='q₂ (y)', linewidth=1.5)
        self.lines['q3'], = ax.plot([], [], color=colors['b'], label='q₃ (z)', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Quaternion')
        ax.set_title('Attitude Quaternion')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        
        # Control - will be initialized on first update with actual n_controls
        ax = self.axes['control']
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control (normalized)')
        ax.set_title('Control Inputs (±1 = limit)')
        ax.grid(True, alpha=0.3)
        
        # Angle error
        ax = self.axes['angle_error']
        self.lines['angle'], = ax.plot([], [], color=colors['b'], linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle Error (deg)')
        ax.set_title('Pointing Error')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 180)
        
        # Convergence
        ax = self.axes['convergence']
        self.lines['cost'], = ax.semilogy([], [], color=colors['b'], label='Cost', linewidth=1.5)
        self.lines['cmax'], = ax.semilogy([], [], color=colors['r'], label='Max Violation', linewidth=1.5)
        self.lines['grad'], = ax.semilogy([], [], color=colors['g'], label='Gradient', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Convergence')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add iteration text
        self.texts['iteration'] = self.fig.text(
            0.5, 0.98, 
            'Iteration: 0 | Outer: 0 | Inner: 0',
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=self.fig.transFigure
        )
        
        self.texts['metrics'] = self.fig.text(
            0.5, 0.95,
            'Cost: -- | Cmax: -- | Grad: --',
            ha='center', va='top', fontsize=10,
            transform=self.fig.transFigure
        )
        
        # Show figure
        self.fig.canvas.draw()
        plt.show(block=False)
        self.fig.canvas.flush_events()
        
        self._is_started = True
    
    def _init_control_lines(self, n_controls: int) -> None:
        """Initialize control line objects for the given number of controls."""
        if self._controls_initialized:
            return
            
        ax = self.axes['control']
        colors = get_control_colors(n_controls)
        labels = get_control_labels(n_controls, self.actuator_names)
        
        # Create a line for each control channel
        for i in range(n_controls):
            line, = ax.plot([], [], color=colors[i], label=labels[i], linewidth=1.2)
            self.lines[f'u{i}'] = line
        
        # Update legend
        ax.legend(loc='upper right', fontsize=7, ncol=max(1, n_controls // 4))
        
        self.n_controls = n_controls
        self._controls_initialized = True
        
    def update(self, iter_data) -> None:
        """
        Update visualization with new iteration data.
        
        This is designed to be used as a callback for PythonALILQR.
        
        Parameters
        ----------
        iter_data : IterationData
            Iteration data from PythonALILQR
        """
        if not self._is_started:
            self.start()
            
        self.iteration_count += 1
        
        # Skip updates based on interval
        if self.iteration_count % self.update_interval != 0:
            return
        
        try:
            self._update_plots(iter_data)
        except Exception as e:
            import traceback
            warnings.warn(f"Visualization update failed: {e}\n{traceback.format_exc()}")
    
    def _update_plots(self, iter_data) -> None:
        """Internal method to update all plots."""
        Xset = iter_data.Xset
        Uset = iter_data.Uset
        N = Xset.shape[1]
        n_ctrl = Uset.shape[0]
        
        # Compute dt based on trajectory - assume total duration is N_base * dt_base
        # where N_base is the original Pass1 size
        if self.goal_vector_eci is not None and self.goal_vector_eci.ndim == 2:
            # We have time-varying goals - use their length to infer total duration
            N_goals = self.goal_vector_eci.shape[1]
            total_duration = (N_goals - 1) * self.dt
            dt_actual = total_duration / (N - 1) if N > 1 else self.dt
        else:
            dt_actual = self.dt
        times = np.arange(N) * dt_actual
        
        # Initialize control lines if needed
        if not self._controls_initialized:
            self._init_control_lines(n_ctrl)
        
        # Update convergence history
        self.cost_history.append(iter_data.LA)
        self.cmax_history.append(max(iter_data.cmax, 1e-16))  # Avoid log(0)
        self.grad_history.append(max(iter_data.grad, 1e-16))
        
        # Angular velocity
        self.lines['omega_x'].set_data(times, Xset[0, :])
        self.lines['omega_y'].set_data(times, Xset[1, :])
        self.lines['omega_z'].set_data(times, Xset[2, :])
        ax = self.axes['omega']
        ax.relim()
        ax.autoscale_view()
        
        # Quaternion
        self.lines['q0'].set_data(times, Xset[3, :])
        self.lines['q1'].set_data(times, Xset[4, :])
        self.lines['q2'].set_data(times, Xset[5, :])
        self.lines['q3'].set_data(times, Xset[6, :])
        self.axes['quat'].set_xlim(0, times[-1])
        
        # Control - handle all channels
        ctrl_len = Uset.shape[1]
        ctrl_times = times[:ctrl_len]
        
        for i in range(n_ctrl):
            key = f'u{i}'
            if key in self.lines:
                # Normalize control by umax if available
                if self.umax is not None and i < len(self.umax):
                    ctrl_normalized = Uset[i, :] / self.umax[i]
                else:
                    ctrl_normalized = Uset[i, :]
                self.lines[key].set_data(ctrl_times, ctrl_normalized)
        
        ax = self.axes['control']
        # Set x-limits to match time
        ax.set_xlim(0, times[-1])
        # Fixed y-limits: +/- 5x the normalized max (which is 1.0)
        if self.umax is not None:
            ax.set_ylim(-5, 5)
        else:
            ax.relim()
            ax.autoscale_view()
        
        # Angle error
        if self.goal_vector_eci is not None:
            goal_vec = self.goal_vector_eci
            # Resample time-varying goals if size doesn't match trajectory
            if goal_vec.ndim == 2 and goal_vec.shape[1] != N:
                # Interpolate goal vectors to match trajectory length
                old_N = goal_vec.shape[1]
                old_times = np.linspace(0, 1, old_N)
                new_times = np.linspace(0, 1, N)
                n_rows = goal_vec.shape[0]  # Preserve dimension (3 for vector, 4 for quaternion)
                goal_vec = np.zeros((n_rows, N))
                for i in range(n_rows):
                    goal_vec[i, :] = np.interp(new_times, old_times, self.goal_vector_eci[i, :])
            angles = compute_angle_error(Xset, goal_vec, self.body_vector)
            self.lines['angle'].set_data(times, angles)
            ax = self.axes['angle_error']
            max_angle = np.nanmax(angles) if not np.all(np.isnan(angles)) else 180
            ax.set_ylim(0, max(max_angle * 1.1, 10))
            ax.set_xlim(0, times[-1])
        else:
            # Show quaternion deviation from identity if no goal specified
            q_err = np.abs(Xset[3, :] - 1.0) + np.sum(np.abs(Xset[4:7, :]), axis=0)
            self.lines['angle'].set_data(times, np.degrees(2 * np.arcsin(np.clip(q_err/2, 0, 1))))
            ax = self.axes['angle_error']
            ax.relim()
            ax.autoscale_view()
        
        # Convergence
        iters = np.arange(len(self.cost_history))
        self.lines['cost'].set_data(iters, self.cost_history)
        self.lines['cmax'].set_data(iters, self.cmax_history)
        self.lines['grad'].set_data(iters, self.grad_history)
        ax = self.axes['convergence']
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(0, max(len(self.cost_history), 1))
        
        # Update text - include pass label if available
        pass_str = f"[{iter_data.pass_label}] " if hasattr(iter_data, 'pass_label') and iter_data.pass_label else ""
        self.texts['iteration'].set_text(
            f'{pass_str}Iteration: {self.iteration_count} | '
            f'Outer: {iter_data.outer_iter} | '
            f'Inner: {iter_data.inner_iter}'
        )
        
        self.texts['metrics'].set_text(
            f'Cost: {iter_data.LA:.4e} | '
            f'Cmax: {iter_data.cmax:.4e} | '
            f'Grad: {iter_data.grad:.4e} | '
            f'μ: {iter_data.mu:.2e} | '
            f'ρ: {iter_data.rho:.2e}'
        )
        
        # Flush updates
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def finish(self, block: bool = True) -> None:
        """
        Finish visualization and optionally block.
        
        Parameters
        ----------
        block : bool
            If True, block until figure is closed
        """
        if self.fig is not None:
            plt.ioff()
            if block:
                plt.show(block=True)
                
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {filename}")
            
    def reset(self) -> None:
        """Reset the visualization for a new optimization run."""
        self.iteration_count = 0
        self.cost_history = []
        self.cmax_history = []
        self.grad_history = []
        
        # Reset control initialization so it can adapt to new problem
        self._controls_initialized = False
        
        if self._is_started:
            # Clear all lines
            for key, line in list(self.lines.items()):
                if key.startswith('u') and key != 'u1' and key != 'u2' and key != 'u3':
                    # Remove dynamically added control lines
                    line.remove()
                    del self.lines[key]
                else:
                    line.set_data([], [])
            
            # Clear control axes legend
            self.axes['control'].legend().remove() if self.axes['control'].get_legend() else None
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


def create_live_callback(
    goal_vector_eci: Optional[NDArray] = None,
    body_vector: Optional[NDArray] = None,
    dt: float = 1.0,
    update_interval: int = 1,
    **kwargs
) -> Tuple[LivePlannerViz, Callable]:
    """
    Convenience function to create a visualization and callback.
    
    Returns
    -------
    viz : LivePlannerViz
        The visualization object
    callback : callable
        Callback function to pass to PythonALILQR
        
    Example
    -------
    >>> viz, callback = create_live_callback(goal_vector_eci=np.array([0, 0, 1]))
    >>> py_alilqr = PythonALILQR(planner, debug_callback=callback)
    >>> result = py_alilqr.optimize(...)
    >>> viz.finish()
    """
    viz = LivePlannerViz(
        goal_vector_eci=goal_vector_eci,
        body_vector=body_vector,
        dt=dt,
        update_interval=update_interval,
        **kwargs
    )
    viz.start()
    
    return viz, viz.update


class ConvergenceMonitor:
    """
    Simple text-based convergence monitor for headless environments.
    
    Prints progress without requiring matplotlib.
    """
    
    def __init__(self, print_interval: int = 5):
        self.print_interval = print_interval
        self.iteration_count = 0
        self.best_cost = float('inf')
        self.best_cmax = float('inf')
        
    def update(self, iter_data) -> None:
        """Update callback for PythonALILQR."""
        self.iteration_count += 1
        
        # Track best values
        if iter_data.LA < self.best_cost:
            self.best_cost = iter_data.LA
        if iter_data.cmax < self.best_cmax:
            self.best_cmax = iter_data.cmax
        
        if self.iteration_count % self.print_interval == 0:
            print(f"[{self.iteration_count:4d}] "
                  f"outer={iter_data.outer_iter:2d} "
                  f"inner={iter_data.inner_iter:2d} | "
                  f"cost={iter_data.LA:10.4e} "
                  f"cmax={iter_data.cmax:10.4e} "
                  f"grad={iter_data.grad:10.4e} | "
                  f"μ={iter_data.mu:.2e}")
            
    def summary(self) -> None:
        """Print final summary."""
        print(f"\n{'='*60}")
        print(f"Optimization complete: {self.iteration_count} iterations")
        print(f"Best cost: {self.best_cost:.6e}")
        print(f"Best cmax: {self.best_cmax:.6e}")
        print(f"{'='*60}")

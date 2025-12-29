import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional, Union
from tqdm import tqdm

# Import necessary ADCS classes
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.CONOPS.goallist import GoalList
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import normalize

def print_header(text):
    print(f"\n{'='*60}\n{text}\n{'='*60}")

class Plan_and_Track_LQR(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> None:
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        
        # System dimensions
        self.nx = est_sat.state_len
        self.nu = est_sat.control_len
        
        # Timing
        self.dt = planner_settings.dt_tvlqr
        self.dt_tp = planner_settings.dt_tp
        
        # Weights Setup
        # Q: State Cost Matrix (nx, nx)
        # R: Control Cost Matrix (nu, nu)
        q_diag = np.concatenate([
            np.full(3, 10.0),    # Angular Velocity
            np.full(4, 100.0),   # Quaternion
            np.full(3, 0.1)      # Wheel Momentum
        ])
        
        # Handle cases where state dimension might differ (e.g. no wheels)
        self.Q = np.diag(q_diag[:self.nx]) 
        self.R = np.eye(self.nu) * 0.01
        
        # Terminal Cost (Qf) - significantly higher to enforce target convergence
        self.Qf = self.Q * 1000.0
        
        self.active_trajectory: Optional[Trajectory] = None

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None) -> np.ndarray:
        """
        Real-time TVLQR tracking controller.
        Calculates u = u_ff - K @ (x - x_ref).
        """
        current_time = os_hat.J2000
        
        if self.active_trajectory is None:
            return np.zeros(self.nu)
        
        if not self.active_trajectory.is_valid_time(current_time):
             return np.zeros(self.nu)
        
        # Interpolate trajectory reference
        x_ref, u_ff, K, _ = self.active_trajectory.get_state_input_gain(current_time)
        
        # State Error
        dx = x_hat - x_ref
        
        # Control Law
        u = u_ff - K @ dx
        
        # Actuator Saturation
        u = np.clip(u, -self.planner_settings.umax, self.planner_settings.umax)
        
        return u

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def calculate_trajectory(self, t_start: float, duration: float, x_0: np.ndarray, os_0: Orbital_State, goals: GoalList, verbose: bool = False) -> Trajectory:
        """
        Generates a trajectory using ALTRO (Augmented Lagrangian Trajectory Optimization).
        Uses a boxed-LQR (iLQR with projected Newton) approach for speed.
        """
        if verbose: 
            print_header(f"ALTRO Trajectory Planner\nStart: {t_start:.5f} | Duration: {duration}s | dt: {self.dt}s")
        
        dt = self.dt
        N = int(np.ceil(duration / dt)) + 1
        t_end = t_start + (duration * TimeConstants.sec2cent)
        
        # 1. Environment Propagation
        #    Pre-calculate Orbital_States for the entire horizon to avoid overhead in loops.
        if verbose: print(">> Propagating Environment...")
        os_list = self._propagate_environment(os_0, t_start, t_end, dt, N)
        times = np.array([os.J2000 for os in os_list])
        
        # 2. Initial Guess
        #    Initialize with zero control or dampening.
        #    Data Format: X (N, nx), U (N-1, nu)
        if verbose: print(">> Generating Initial Rollout...")
        U_guess = np.zeros((N-1, self.nu))
        X_guess = self._rollout(x_0, U_guess, dt, os_list)
        
        # 3. Solver Loop
        if verbose: print(">> Starting Optimization Loop...")
        X_opt, U_opt, K_opt, P_opt = self._altro_solve(X_guess, U_guess, dt, os_list, goals, verbose)
        
        # 4. Packaging
        #    K_opt is (N-1, nu, nx). 
        #    Trajectory class likely expects K aligned with time steps.
        #    We append a zero gain for the final step N if necessary for consistency.
        K_full = np.concatenate([K_opt, np.zeros((1, self.nu, self.nx))], axis=0)
        
        return Trajectory(times, X_opt, U_opt, K_full, P_opt)

    def _altro_solve(self, X_init, U_init, dt, os_list, goals, verbose=False):
        """
        The Core iLQR / ALTRO Loop with Levenberg-Marquardt Regularization.
        """
        max_iter = 50
        tol_cost = 1e-7
        
        # Regularization parameters
        reg = 1e-6
        reg_min = 1e-6
        reg_max = 1e9
        reg_factor = 10.0
        
        X = X_init.copy()
        U = U_init.copy()
        
        current_cost = self._trajectory_cost(X, U, goals)
        
        if verbose:
            print(f"{'Iter':<5} | {'Cost':<12} | {'dCost':<12} | {'Alpha':<8} | {'Reg':<8}")
            print("-" * 60)

        N = U.shape[0] + 1
        Ks = np.zeros((N-1, self.nu, self.nx))
        
        for iteration in range(max_iter):
            
            # --- A. Backward Pass ---
            # Pass 'reg' to backward pass
            Ks, ds, delta_V = self._backward_pass(X, U, dt, os_list, goals, reg)
            
            # --- B. Forward Pass ---
            X_new, U_new, new_cost, alpha = self._forward_pass(X, U, Ks, ds, dt, os_list, goals, current_cost)
            
            d_cost = current_cost - new_cost
            
            # --- C. Regularization Update ---
            if d_cost > 0:
                # Step accepted: Decrease regularization (trust the model more)
                if verbose:
                    print(f"{iteration:<5} | {new_cost:.6e} | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e}")
                
                d_cost_rel = d_cost / np.abs(current_cost)
                X = X_new
                U = U_new
                current_cost = new_cost
                
                # Update Reg: Decrease it (trust region expansion)
                reg = max(reg_min, reg / reg_factor)
                
                # MODIFIED CHECK: Only stop if improvement is small AND we aren't heavily regularized
                if d_cost_rel < tol_cost and reg < 1.0: 
                    if verbose: print(">> Converged (Cost Tolerance).")
                    break
            else:
                # Step rejected: Increase regularization (trust the model less)
                # We keep the old X/U and try again with higher damping
                if verbose:
                    print(f"{iteration:<5} | REJECTED     | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e}")
                
                reg = min(reg_max, reg * reg_factor)
                if reg >= reg_max:
                    if verbose: print(">> Converged (Regularization Limit).")
                    break

        P_final = np.zeros((N, self.nx, self.nx)) 
        return X, U, Ks, P_final

    def _backward_pass(self, X: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State], goals: GoalList, reg: float):
        """
        Computes optimal gains K and feedforward d by solving the Riccati equation backwards.
        Includes Regularization (Levenberg-Marquardt) to ensure positive definiteness.
        """
        N = X.shape[0]
        
        Ks = np.zeros((N-1, self.nu, self.nx))
        ds = np.zeros((N-1, self.nu))
        
        # 1. Terminal Cost Derivatives
        x_goal = np.zeros(self.nx) 
        
        Vx = self.Qf @ (X[-1] - x_goal)
        Vxx = self.Qf.copy()
        
        delta_V = 0.0
        
        # 2. Backward Loop
        for k in range(N-2, -1, -1):
            x_k = X[k]
            u_k = U[k]
            os_k = os_list[k]
            
            # --- Linearization ---
            jacobians = self.est_sat.dynJacCore(x_k, u_k, os_k)
            A_c = jacobians[0] # (nx, nx)
            B_c_T = jacobians[1] # (nu, nx)
            
            # Discrete Jacobians (Euler approx)
            A_k = np.eye(self.nx) + A_c * dt
            B_k = B_c_T.T * dt
            
            # --- Quadratic Approximation of Cost ---
            lx = self.Q @ (x_k - x_goal)
            lu = self.R @ u_k
            lxx = self.Q
            luu = self.R
            
            # --- Q-Function Terms ---
            Qx  = lx + A_k.T @ Vx
            Qu  = lu + B_k.T @ Vx
            
            Qxx = lxx + A_k.T @ Vxx @ A_k
            Quu = luu + B_k.T @ Vxx @ B_k
            Qux = B_k.T @ Vxx @ A_k
            
            # --- Solve for Gains with Regularization ---
            # Apply Levenberg-Marquardt regularization
            Quu_reg = Quu + np.eye(self.nu) * reg
            
            # Use Cholesky for speed and stability check
            try:
                # Solve (Quu + reg*I) * [d, K] = -[Qu, Qux]
                # Lower triangular Cholesky Factor
                L = scipy.linalg.cho_factor(Quu_reg, lower=True)
                
                # Feedback Gain K
                K = -scipy.linalg.cho_solve(L, Qux)
                # Feedforward d
                d = -scipy.linalg.cho_solve(L, Qu)
                
            except scipy.linalg.LinAlgError:
                # If Cholesky fails (not PD), fallback to Eigen decomposition or pseudoinverse
                # This usually only happens if reg is too small, which the outer loop handles.
                Quu_evals, Quu_evecs = np.linalg.eigh(Quu_reg)
                Quu_evals[Quu_evals < 1e-6] = 1e-6
                Quu_inv = Quu_evecs @ np.diag(1.0 / Quu_evals) @ Quu_evecs.T
                
                K = -Quu_inv @ Qux
                d = -Quu_inv @ Qu
            
            Ks[k] = K
            ds[k] = d
            
            # --- Update Value Function (V) ---
            # Vx = Qx + K' Quu d + K' Qu + Qux' d
            # Vxx = Qxx + K' Quu K + K' Qux + Qux' K
            
            Vx = Qx + K.T @ Quu @ d + K.T @ Qu + Qux.T @ d
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            # Ensure symmetry
            Vxx = 0.5 * (Vxx + Vxx.T)
            
        return Ks, ds, delta_V

    def _forward_pass(self, X_old, U_old, Ks, ds, dt, os_list, goals, current_cost):
        """
        Rollout with Line Search (Backtracking).
        Includes checks for divergence (NaN/Inf) to prevent crashes.
        """
        N = U_old.shape[0] + 1
        # Try smaller steps first to avoid violent divergence
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.01, 0.001]
        
        best_X = X_old
        best_U = U_old
        best_cost = current_cost
        best_alpha = 0.0
        
        for alpha in alphas:
            X_new = np.zeros_like(X_old)
            U_new = np.zeros_like(U_old)
            X_new[0] = X_old[0]
            
            diverged = False
            
            # Rollout
            for k in range(N-1):
                x_k = X_new[k]
                
                # Divergence Check: If state is exploding, abort this alpha immediately
                if not np.all(np.isfinite(x_k)) or np.max(np.abs(x_k)) > 1e10:
                    diverged = True
                    break

                delta_x = x_k - X_old[k]
                delta_u = alpha * ds[k] + Ks[k] @ delta_x
                
                u_applied = np.clip(U_old[k] + delta_u, -self.planner_settings.umax, self.planner_settings.umax)
                U_new[k] = u_applied
                
                # Dynamics Step
                # We wrap this in try-except to catch RuntimeWarnings promoted to errors
                try:
                    X_new[k+1] = self._rk4_step(x_k, u_applied, dt, os_list[k], os_list[k+1])
                except (RuntimeWarning, ValueError, OverflowError):
                    diverged = True
                    break

            if diverged:
                # This alpha was too big, try the next smaller one
                continue

            # Calculate Cost
            cost = self._trajectory_cost(X_new, U_new, goals)
            
            # Acceptance Criteria
            if cost < current_cost:
                return X_new, U_new, cost, alpha
        
        # If all alphas failed, return original trajectory (zero improvement)
        return X_old, U_old, current_cost, 0.0

    def _trajectory_cost(self, X: np.ndarray, U: np.ndarray, goals: GoalList) -> float:
        """
        Calculates total trajectory cost.
        """
        # Note: Ideally, x_goal should be retrieved from GoalList for every time step
        x_goal = np.zeros(self.nx)
        
        J = 0.0
        N = X.shape[0]
        
        # Stage Cost
        for k in range(N-1):
            dx = X[k] - x_goal
            u = U[k]
            # 0.5 * (x'Qx + u'Ru)
            J += 0.5 * (dx.T @ self.Q @ dx + u.T @ self.R @ u)
            
        # Terminal Cost
        dx_N = X[-1] - x_goal
        J += 0.5 * dx_N.T @ self.Qf @ dx_N
        
        return J

    def _rollout(self, x0: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State]) -> np.ndarray:
        """
        Simulates dynamics forward for the entire horizon.
        """
        N = U.shape[0] + 1
        X = np.zeros((N, self.nx))
        X[0] = x0
        
        for k in range(N-1):
            X[k+1] = self._rk4_step(X[k], U[k], dt, os_list[k], os_list[k+1])
            
        return X

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float, os_prev: Orbital_State, os_next: Orbital_State) -> np.ndarray:
        """
        RK4 Integration Step compatible with Satellite.dynamics_for_solver.
        """
        # dynamics_for_solver expects: (t, x, u, os0, os1)
        # It internally interpolates between os0 and os1 based on t.
        # For RK4, we need derivatives at t=0, t=dt/2, t=dt.
        
        # k1 @ t=0
        k1 = self.est_sat.dynamics_for_solver(0.0, x, u, os_prev, os_next)
        
        # k2 @ t=dt/2
        k2 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k1, u, os_prev, os_next)
        
        # k3 @ t=dt/2
        k3 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k2, u, os_prev, os_next)
        
        # k4 @ t=dt
        k4 = self.est_sat.dynamics_for_solver(dt, x + dt*k3, u, os_prev, os_next)
        
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize Quaternion (indices 3:7)
        q = x_next[3:7]
        norm_sq = np.dot(q, q)
        if norm_sq > 1e-12:
            x_next[3:7] /= np.sqrt(norm_sq)
            
        return x_next

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt: float, N: int) -> List[Orbital_State]:
        """
        Propagates the orbit and generates Orbital_State objects for every time step.
        Uses fast=True to skip heavy Skyfield frame conversions.
        """
        # Buffer end time slightly for Orbit generator
        orb = Orbit(os0=os_0, end_time=t_end + 10*dt*TimeConstants.sec2cent, dt=dt, fast=False)
        
        # Get raw vectors (List[List[float]]) or Arrays
        tp_orbit = orb.get_range(t_start, t_end, dt)
        vecs_raw = tp_orbit.get_vecs() 
        times = np.array(tp_orbit.times)
        
        # Unpack raw data into Numpy Arrays and slice/pad to N
        # Standard Orbit.get_vecs order: [R, V, B, S, Rho]
        R = np.array(vecs_raw[0])
        V = np.array(vecs_raw[1])
        B = np.array(vecs_raw[2])
        S = np.array(vecs_raw[3])
        
        # Handle Length Mismatch
        current_len = len(R)
        if current_len < N:
            pad = N - current_len
            # Pad with last value (Zero order hold)
            R = np.pad(R, ((0,pad), (0,0)), 'edge')
            V = np.pad(V, ((0,pad), (0,0)), 'edge')
            B = np.pad(B, ((0,pad), (0,0)), 'edge')
            S = np.pad(S, ((0,pad), (0,0)), 'edge')
            times = np.pad(times, (0, pad), 'edge')
        elif current_len > N:
            R = R[:N]
            V = V[:N]
            B = B[:N]
            S = S[:N]
            times = times[:N]

        # Reuse ephemeris to save memory
        ephem = os_0.ephem
        os_list = []
        
        for i in range(N):
            # Create fast Orbital_State with pre-computed vectors
            os = Orbital_State(
                ephem=ephem,
                J2000=times[i],
                R=R[i],
                V=V[i],
                B=B[i],
                S=S[i],
                rho=0.0, 
                density_model=None,
                fast=True 
            )
            os_list.append(os)
        
        return os_list
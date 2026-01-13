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
        
        # --- Weights Setup ---
        # Q: State Cost Matrix (nx, nx)
        q_diag = np.concatenate([
            np.full(3, 10.0),    # Angular Velocity
            np.full(4, 50.0),    # Quaternion (Reduced slightly to prevent aggressive initial snaps)
            np.full(3, 0.1)      # Wheel Momentum
        ])
        
        self.Q = np.diag(q_diag[:self.nx]) 
        
        # --- CRITICAL FIX: Higher R to prevent actuator saturation ---
        # If max torque is ~0.005, and we want u^T R u to be comparable to x^T Q x,
        # R needs to be high. 
        # Example: Cost of 0.005 Nm should be substantial.
        self.R = np.eye(self.nu) * 1000.0 
        
        # Terminal Cost (Qf)
        self.Qf = self.Q * 100.0
        
        self.active_trajectory: Optional[Trajectory] = None

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None) -> np.ndarray:
        current_time = os_hat.J2000
        
        if self.active_trajectory is None:
            return np.zeros(self.nu)
        
        if not self.active_trajectory.is_valid_time(current_time):
             return np.zeros(self.nu)
        
        # Interpolate trajectory reference
        x_ref, u_ff, K, _ = self.active_trajectory.get_state_input_gain(current_time)
        
        if np.dot(x_ref[3:7], x_ref[3:7]) < 1e-6:
            x_ref[3:7] = np.array([1.0, 0, 0, 0])

        dx = x_hat - x_ref
        
        # Quaternion Shortest Path
        if self.nx >= 7 and np.dot(x_hat[3:7], x_ref[3:7]) < 0:
            dx[3:7] = x_hat[3:7] + x_ref[3:7]

        u = u_ff - K @ dx
        return u

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def calculate_trajectory(self, t_start: float, duration: float, x_0: np.ndarray, os_0: Orbital_State, goals: GoalList, verbose: bool = False) -> Trajectory:
        if verbose: 
            print_header(f"ALTRO Trajectory Planner\nStart: {t_start:.5f} | Duration: {duration}s | dt: {self.dt}s")
        
        dt = self.dt
        N = int(np.ceil(duration / dt)) + 1
        t_end = t_start + (duration * TimeConstants.sec2cent)
        
        if verbose: print(">> Propagating Environment...")
        os_list = self._propagate_environment(os_0, t_start, t_end, dt, N)
        times = np.array([os.J2000 for os in os_list])
        
        if verbose: print(">> Generating Initial Rollout...")
        U_guess = np.zeros((N-1, self.nu))
        X_guess = self._rollout(x_0, U_guess, dt, os_list)
        
        if verbose: print(">> Starting Optimization Loop...")
        X_opt, U_opt, K_opt, P_opt = self._altro_solve(X_guess, U_guess, dt, os_list, goals, verbose)
        
        K_full = np.concatenate([K_opt, np.zeros((1, self.nu, self.nx))], axis=0)
        
        return Trajectory(times, X_opt, U_opt, K_full, P_opt)

    def _altro_solve(self, X_init, U_init, dt, os_list, goals, verbose=False):
        max_iter = 50
        tol_cost = 1e-5
        
        reg = 1e-6
        reg_min = 1e-8
        reg_max = 1e9
        reg_factor = 5.0  # Reduced factor for smoother updates
        
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
            try:
                Ks, ds, delta_V = self._backward_pass(X, U, dt, os_list, goals, reg)
            except np.linalg.LinAlgError:
                # If backward pass fails (matrix inversion), increase reg and retry
                reg = min(reg_max, reg * reg_factor * 10)
                if verbose: print(f"{iteration:<5} | BACKWARD ERR | -            | -        | {reg:.1e}")
                continue

            # --- B. Forward Pass ---
            X_new, U_new, new_cost, alpha = self._forward_pass(X, U, Ks, ds, dt, os_list, goals, current_cost)
            
            d_cost = current_cost - new_cost
            
            # --- C. Regularization Update ---
            if d_cost > 0:
                if verbose:
                    print(f"{iteration:<5} | {new_cost:.6e} | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e}")
                
                d_cost_rel = d_cost / (np.abs(current_cost) + 1e-6)
                X = X_new
                U = U_new
                current_cost = new_cost
                
                reg = max(reg_min, reg / reg_factor)
                
                if d_cost_rel < tol_cost: 
                    if verbose: print(">> Converged (Cost Tolerance).")
                    break
            else:
                if verbose:
                    print(f"{iteration:<5} | REJECTED     | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e}")
                
                reg = min(reg_max, reg * reg_factor)
                if reg >= reg_max:
                    if verbose: print(">> Converged (Regularization Limit).")
                    break

        P_final = np.zeros((N, self.nx, self.nx)) 
        return X, U, Ks, P_final

    def _backward_pass(self, X: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State], goals: GoalList, reg: float):
        N = X.shape[0]
        
        Ks = np.zeros((N-1, self.nu, self.nx))
        ds = np.zeros((N-1, self.nu))
        
        # --- Target Setup ---
        # NOTE: Ideally this extracts from 'goals', but using Identity for stability as per user code
        x_goal = np.zeros(self.nx)
        x_goal[3] = 1.0 # Identity Quaternion (scalar first [w, x, y, z]) assumption

        dx = X[-1] - x_goal
        if self.nx >= 7 and np.dot(X[-1, 3:7], x_goal[3:7]) < 0:
            dx[3:7] = X[-1, 3:7] + x_goal[3:7]

        Vx = self.Qf @ dx
        Vxx = self.Qf.copy()
        
        delta_V = 0.0
        
        for k in range(N-2, -1, -1):
            x_k = X[k]
            u_k = U[k]
            os_k = os_list[k]
            
            # Linearization (Continuous)
            jacobians = self.est_sat.dynJacCore(x_k, u_k, os_k)
            A_c = jacobians[0]
            B_c_T = jacobians[1]
            
            # --- CRITICAL FIX: 2nd Order Discretization ---
            # Approximates RK4/Matrix Exp much better than Euler for dt=1.0
            # A_k = I + A*dt + 0.5*A^2*dt^2
            # B_k = B*dt + 0.5*A*B*dt^2
            
            A_sq = A_c @ A_c
            A_k = np.eye(self.nx) + A_c * dt + 0.5 * A_sq * (dt**2)
            B_k = B_c_T.T * dt + 0.5 * (A_c @ B_c_T.T) * (dt**2)
            
            lx = self.Q @ (x_k - x_goal)
            lu = self.R @ u_k
            lxx = self.Q
            luu = self.R
            
            Qx  = lx + A_k.T @ Vx
            Qu  = lu + B_k.T @ Vx
            
            Qxx = lxx + A_k.T @ Vxx @ A_k
            Quu = luu + B_k.T @ Vxx @ B_k
            Qux = B_k.T @ Vxx @ A_k
            
            # Regularization
            Quu_reg = Quu + np.eye(self.nu) * reg
            
            try:
                L = scipy.linalg.cho_factor(Quu_reg, lower=True)
                K = -scipy.linalg.cho_solve(L, Qux)
                d = -scipy.linalg.cho_solve(L, Qu)
            except scipy.linalg.LinAlgError:
                Quu_evals, Quu_evecs = np.linalg.eigh(Quu_reg)
                Quu_evals[Quu_evals < 1e-6] = 1e-6
                Quu_inv = Quu_evecs @ np.diag(1.0 / Quu_evals) @ Quu_evecs.T
                K = -Quu_inv @ Qux
                d = -Quu_inv @ Qu
            
            Ks[k] = K
            ds[k] = d
            
            Vx = Qx + K.T @ Quu @ d + K.T @ Qu + Qux.T @ d
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)
            
        return Ks, ds, delta_V

    def _forward_pass(self, X_old, U_old, Ks, ds, dt, os_list, goals, current_cost):
        N = U_old.shape[0] + 1
        alphas = [1.0, 0.5, 0.25, 0.125, 0.05, 0.01]
        
        for alpha in alphas:
            X_new = np.zeros_like(X_old)
            U_new = np.zeros_like(U_old)
            X_new[0] = X_old[0]
            
            diverged = False
            
            for k in range(N-1):
                x_k = X_new[k]
                
                if not np.all(np.isfinite(x_k)) or np.max(np.abs(x_k)) > 1e10:
                    diverged = True
                    break

                delta_x = x_k - X_old[k]
                
                # Check for quaternion flip in delta calculation
                if self.nx >= 7 and np.dot(x_k[3:7], X_old[k, 3:7]) < 0:
                    delta_x[3:7] = x_k[3:7] + X_old[k, 3:7]

                delta_u = alpha * ds[k] + Ks[k] @ delta_x
                
                u_applied = np.clip(U_old[k] + delta_u, -self.planner_settings.umax, self.planner_settings.umax)
                U_new[k] = u_applied
                
                try:
                    X_new[k+1] = self._rk4_step(x_k, u_applied, dt, os_list[k], os_list[k+1])
                except (RuntimeWarning, ValueError, OverflowError):
                    diverged = True
                    break

            if diverged:
                continue

            cost = self._trajectory_cost(X_new, U_new, goals)
            
            if cost < current_cost:
                return X_new, U_new, cost, alpha
        
        return X_old, U_old, current_cost, 0.0

    def _trajectory_cost(self, X: np.ndarray, U: np.ndarray, goals: GoalList) -> float:
        J = 0.0
        N = X.shape[0]
        x_target_default = np.zeros(self.nx)
        x_target_default[3] = 1.0 

        for k in range(N-1):
            dx = X[k] - x_target_default
            if self.nx >= 7 and np.dot(X[k, 3:7], x_target_default[3:7]) < 0:
                dx[3:7] = X[k, 3:7] + x_target_default[3:7]

            u = U[k]
            # Use self.R here to ensure consistency with Backward Pass
            J += 0.5 * (dx.T @ self.Q @ dx + u.T @ self.R @ u)
            
        dx_N = X[-1] - x_target_default
        if self.nx >= 7 and np.dot(X[-1, 3:7], x_target_default[3:7]) < 0:
            dx_N[3:7] = X[-1, 3:7] + x_target_default[3:7]

        J += 0.5 * dx_N.T @ self.Qf @ dx_N
        
        return J

    def _rollout(self, x0: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State]) -> np.ndarray:
        N = U.shape[0] + 1
        X = np.zeros((N, self.nx))
        X[0] = x0
        for k in range(N-1):
            X[k+1] = self._rk4_step(X[k], U[k], dt, os_list[k], os_list[k+1])
        return X

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float, os_prev: Orbital_State, os_next: Orbital_State) -> np.ndarray:
        k1 = self.est_sat.dynamics_for_solver(0.0, x, u, os_prev, os_next)
        k2 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k1, u, os_prev, os_next)
        k3 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k2, u, os_prev, os_next)
        k4 = self.est_sat.dynamics_for_solver(dt, x + dt*k3, u, os_prev, os_next)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        q = x_next[3:7]
        norm_sq = np.dot(q, q)
        if norm_sq > 1e-12:
            x_next[3:7] /= np.sqrt(norm_sq)
        return x_next

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt: float, N: int) -> List[Orbital_State]:
        orb = Orbit(os0=os_0, end_time=t_end + 10*dt*TimeConstants.sec2cent, dt=dt, fast=False)
        tp_orbit = orb.get_range(t_start, t_end, dt)
        vecs_raw = tp_orbit.get_vecs() 
        times = np.array(tp_orbit.times)
        
        R = np.array(vecs_raw[0])
        V = np.array(vecs_raw[1])
        B = np.array(vecs_raw[2])
        S = np.array(vecs_raw[3])
        
        current_len = len(R)
        if current_len < N:
            pad = N - current_len
            R = np.pad(R, ((0,pad), (0,0)), 'edge')
            V = np.pad(V, ((0,pad), (0,0)), 'edge')
            B = np.pad(B, ((0,pad), (0,0)), 'edge')
            S = np.pad(S, ((0,pad), (0,0)), 'edge')
            times = np.pad(times, (0, pad), 'edge')
        elif current_len > N:
            R = R[:N]; V = V[:N]; B = B[:N]; S = S[:N]; times = times[:N]

        ephem = os_0.ephem
        os_list = []
        for i in range(N):
            os = Orbital_State(
                ephem=ephem, J2000=times[i], R=R[i], V=V[i], B=B[i], S=S[i],
                rho=0.0, density_model=None, fast=True 
            )
            os_list.append(os)
        return os_list
    
    def _get_reference_state(self, t: float, os: Orbital_State, goals: GoalList) -> np.ndarray:
        # 1. Get Target Vector (ECI) and Target Angular Velocity from GoalList
        r_target_eci, w_target = goals.to_ref(t, os)
        
        # 2. Normalize vectors
        u = normalize(r_target_eci)          # Target in ECI
        v = normalize(self.est_sat.boresight)# Boresight in Body
        
        # 3. Compute Quaternion (ECI -> Body)
        # We want Rotation R(q) such that: R(q) * u_eci = v_body
        # This is the rotation from u to v.
        dot_prod = np.dot(u, v)
        cross_prod = np.cross(u, v)
        
        # Shortest arc quaternion construction
        # q = [cos(theta/2), axis * sin(theta/2)]
        # Using identity: q_w = sqrt(|u|^2 * |v|^2) + dot(u, v)
        q_w = 1.0 + dot_prod
        q_xyz = cross_prod
        
        # Handle 180 degree singularity (vectors opposite)
        if q_w < 1e-6:
            # arbitrary perpendicular axis
            q_xyz = np.cross(u, np.array([1, 0, 0]))
            if np.linalg.norm(q_xyz) < 1e-6:
                q_xyz = np.cross(u, np.array([0, 1, 0]))
            q_w = 0.0

        q_ref = normalize(np.concatenate(([q_w], q_xyz)))
        
        # 4. Construct Full State Ref [w, q, h]
        x_ref = np.zeros(self.nx)
        x_ref[0:3] = w_target
        x_ref[3:7] = q_ref
        
        # Optional: Keep wheels at 0 momentum or preferred state
        # x_ref[7:] = 0.0 
        
        return x_ref
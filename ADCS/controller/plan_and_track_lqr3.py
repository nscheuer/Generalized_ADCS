import numpy as np
import scipy.linalg
from typing import Tuple, List, Optional, Union, Dict
from tqdm import tqdm

# Import necessary ADCS classes
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.CONOPS.goallist import GoalList
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import rot_mat, drotmatTvecdq, Wmat, skewsym, normalize

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
        
        # --- Cost Matrices (Equivalent to costSettings in C++) ---
        # Q: State Cost Matrix (nx, nx)
        # R: Control Cost Matrix (nu, nu)
        q_diag = np.concatenate([
            np.full(3, 10.0),    # Angular Velocity
            np.full(4, 100.0),   # Quaternion
            np.full(3, 0.1)      # Wheel Momentum
        ])
        
        # Safe sizing for satellites without wheels
        self.Q = np.diag(q_diag[:self.nx]) 
        self.R = np.eye(self.nu) * 0.01
        self.Qf = self.Q * 1000.0 # Terminal cost
        
        self.active_trajectory: Optional[Trajectory] = None

    # ---------------------------------------------------------
    # Public Interface
    # ---------------------------------------------------------

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None) -> np.ndarray:
        current_time = os_hat.J2000
        
        if self.active_trajectory is None or not self.active_trajectory.is_valid_time(current_time):
             return np.zeros(self.nu)
        
        x_ref, u_ff, K, _ = self.active_trajectory.get_state_input_gain(current_time)
        dx = x_hat - x_ref
        
        # Quaternion error handling (shortest path)
        if self.nx >= 7 and np.dot(x_hat[3:7], x_ref[3:7]) < 0:
            dx[3:7] = x_hat[3:7] + x_ref[3:7]

        u = u_ff - K @ dx
        u = np.clip(u, -self.planner_settings.umax, self.planner_settings.umax)
        return u

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def calculate_trajectory(self, t_start: float, duration: float, x_0: np.ndarray, os_0: Orbital_State, goals: GoalList, verbose: bool = False) -> Trajectory:
        """
        Main entry point. Mirrors C++ trajOpt.
        """
        if verbose: 
            print_header(f"ALTRO Trajectory Planner (Python)\nStart: {t_start:.5f} | Duration: {duration}s | dt: {self.dt}s")
        
        dt = self.dt
        # Calculate steps N (C++ uses floor/ceil logic, we enforce N)
        N = int(np.ceil(duration / dt)) + 1
        t_end = t_start + (duration * TimeConstants.sec2cent)
        
        # 1. Propagate Environment (Resampling)
        #    This mirrors 'findVecTimes' in C++. We pre-calculate all OrbitalStates.
        if verbose: print(">> Propagating Environment...")
        os_list = self._propagate_environment(os_0, t_start, t_end, dt, N)
        times = np.array([os.J2000 for os in os_list])
        
        # 2. Initialization Phase (trajOptBefore)
        if verbose: print(">> Generating Initial Trajectory...")
        X_init, U_init = self._trajOptBefore(x_0, dt, N, os_list)
        
        # 3. Optimization Phase (alilqr)
        if verbose: print(">> Starting AL-iLQR Optimization...")
        X_opt, U_opt, K_opt, P_opt = self._alilqr(X_init, U_init, dt, os_list, goals, verbose)
        
        # 4. Packaging (trajOptAfter)
        #    Append zero gain for final step to match N
        K_full = np.concatenate([K_opt, np.zeros((1, self.nu, self.nx))], axis=0)
        
        return Trajectory(times, X_opt, U_opt, K_full, P_opt)

    # ---------------------------------------------------------
    # Core Optimization Loop (Mirrors OldPlanner::alilqr)
    # ---------------------------------------------------------

    def _alilqr(self, X_init: np.ndarray, U_init: np.ndarray, dt: float, os_list: List[Orbital_State], goals: GoalList, verbose: bool = False):
        """
        Augmented Lagrangian iLQR Solver.
        Structure:
          - Outer Loop: Updates Lagrange Multipliers / Penalties (Not fully implemented here, placeholder for AL)
          - Inner Loop: Solves LQR with Regularization
        """
        # Settings mirroring C++ config
        max_outer_iter = 5
        max_inner_iter = 30
        tol_grad = 1e-4
        tol_cost = 1e-4
        
        # Regularization (Levenberg-Marquardt)
        reg = 1e-6
        reg_min = 1e-6
        reg_max = 1e9
        reg_scale = 10.0 # Factor to increase/decrease reg
        
        X = X_init.copy()
        U = U_init.copy()
        
        current_cost = self._cost_function(X, U, goals)
        
        if verbose:
            print(f"{'Iter':<6} | {'Cost':<12} | {'dCost':<12} | {'Alpha':<8} | {'Reg':<8} | {'Grad':<8}")
            print("-" * 75)

        N = X.shape[0]
        Ks = np.zeros((N-1, self.nu, self.nx))
        ds = np.zeros((N-1, self.nu))
        
        # --- Outer Loop (AL) ---
        # Currently just one pass effectively, as we focus on iLQR stability first
        for outer in range(1): 
            
            # --- Inner Loop (iLQR) ---
            for inner in range(max_inner_iter):
                
                # 1. Backward Pass
                #    Returns Gains (K, d) and expected Value reduction (delta_V)
                #    If backward pass fails (non-PD), it increases reg and returns None
                bp_result = self._backward_pass(X, U, dt, os_list, goals, reg)
                
                if bp_result is None:
                    # Matrix inversion failed, increase regularization and retry
                    if verbose: print(f"{inner:<6} | {'BACKWARD_ERR':<12} | {'-':<12} | {'-':<8} | {reg:.1e}")
                    reg = min(reg_max, reg * reg_scale)
                    if reg >= reg_max: break
                    continue
                    
                Ks, ds, delta_V = bp_result
                
                # 2. Forward Pass (Line Search)
                #    Applies gains and integrates dynamics
                X_new, U_new, new_cost, alpha = self._forward_pass(X, U, Ks, ds, dt, os_list, goals, current_cost)
                
                d_cost = current_cost - new_cost
                
                # 3. Regularization Update Logic (Levenberg-Marquardt)
                if d_cost > 0:
                    # Successful step
                    grad_norm = np.mean(np.abs(ds)) # Rough gradient proxy
                    
                    if verbose:
                        print(f"{inner:<6} | {new_cost:.6e} | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e} | {grad_norm:.1e}")
                    
                    # Accept Step
                    X = X_new
                    U = U_new
                    current_cost = new_cost
                    
                    # Decrease Regularization (Trust region expansion)
                    reg = max(reg_min, reg / reg_scale)
                    
                    # Convergence Check
                    if d_cost < tol_cost:
                        if verbose: print(">> Converged (Cost Tolerance).")
                        break
                        
                else:
                    # Failed step (Cost increased)
                    if verbose:
                        print(f"{inner:<6} | {'REJECTED':<12} | {d_cost:.6e} | {alpha:<8.4f} | {reg:.1e}")
                    
                    # Increase Regularization (Trust region shrinking)
                    reg = min(reg_max, reg * reg_scale)
                    if reg >= reg_max:
                        if verbose: print(">> Converged (Max Regularization).")
                        break

        # Placeholder for final P (covariance)
        P_final = np.zeros((N, self.nx, self.nx)) 
        return X, U, Ks, P_final

    # ---------------------------------------------------------
    # Backward Pass (Mirrors OldPlanner::backwardPass)
    # ---------------------------------------------------------

    def _backward_pass(self, X: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State], goals: GoalList, reg: float):
        N = X.shape[0]
        Ks = np.zeros((N-1, self.nu, self.nx))
        ds = np.zeros((N-1, self.nu))
        
        # 1. Terminal Cost
        x_goal = np.zeros(self.nx) # Ideally fetch from goals
        Vx = self.Qf @ (X[-1] - x_goal)
        Vxx = self.Qf.copy()
        
        delta_V = 0.0
        
        # 2. Iterate Backwards
        for k in range(N-2, -1, -1):
            x_k = X[k]
            u_k = U[k]
            os_k = os_list[k]
            
            # --- Linearize Dynamics ---
            # Returns [A_continuous, B_continuous_transposed]
            # Must handle the specific output format of dynJacCore
            jacobians = self._linearize_dynamics_local(x_k, u_k, os_k)
            A_c = jacobians[0]
            B_c_T = jacobians[1] # Satellite.py usually returns dxdot/du as (Control x State)
            
            # Discretize (Euler)
            # A_k = I + A_c * dt
            # B_k = B_c * dt
            A_k = np.eye(self.nx) + A_c * dt
            B_k = B_c_T.T * dt # Transpose to get standard (nx, nu)
            
            # --- Quadratic Cost derivatives ---
            # lx = Q*(x-xg), lu = R*u
            lx = self.Q @ (x_k - x_goal)
            lu = self.R @ u_k
            lxx = self.Q
            luu = self.R
            
            # --- Q-Function ---
            # Qx  = lx + A' Vx
            # Qu  = lu + B' Vx
            Qx  = lx + A_k.T @ Vx
            Qu  = lu + B_k.T @ Vx
            
            # Qxx = lxx + A' Vxx A
            # Quu = luu + B' Vxx B
            # Qux = B' Vxx A
            Qxx = lxx + A_k.T @ Vxx @ A_k
            Quu = luu + B_k.T @ Vxx @ B_k
            Qux = B_k.T @ Vxx @ A_k
            
            # --- Regularization ---
            # Add damping to Quu to ensure invertibility
            Quu_reg = Quu + np.eye(self.nu) * reg
            
            # --- Solve for Gains ---
            try:
                # Cholesky Decomposition for speed/stability
                L = scipy.linalg.cho_factor(Quu_reg, lower=True)
                K = -scipy.linalg.cho_solve(L, Qux)
                d = -scipy.linalg.cho_solve(L, Qu)
            except scipy.linalg.LinAlgError:
                # Matrix not PD -> Increase reg (Triggered by returning None)
                return None
            
            Ks[k] = K
            ds[k] = d
            
            # --- Update Value Function (V) ---
            # Vx = Qx + K' Quu d + K' Qu + Qux' d
            # Vxx = Qxx + K' Quu K + K' Qux + Qux' K
            Vx = Qx + K.T @ Quu @ d + K.T @ Qu + Qux.T @ d
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            # Symmetrize Vxx to prevent numerical drift
            Vxx = 0.5 * (Vxx + Vxx.T)
            
        return Ks, ds, delta_V

    # ---------------------------------------------------------
    # Forward Pass (Mirrors OldPlanner::forwardPass)
    # ---------------------------------------------------------

    def _forward_pass(self, X_old, U_old, Ks, ds, dt, os_list, goals, current_cost):
        N = X_old.shape[0]
        # Backtracking line search
        alphas = [1.0, 0.5, 0.25, 0.125, 0.05, 0.01, 0.001]
        
        for alpha in alphas:
            X_new = np.zeros_like(X_old)
            U_new = np.zeros_like(U_old)
            X_new[0] = X_old[0]
            
            diverged = False
            
            for k in range(N-1):
                x_k = X_new[k]
                
                # Check for NaN/Inf/Explosion
                if not np.all(np.isfinite(x_k)) or np.max(np.abs(x_k)) > 1e6:
                    diverged = True
                    break
                
                # Feedback Control: u = u_old + alpha*d + K(x_new - x_old)
                delta_x = x_k - X_old[k]
                
                # Handle quaternion wrapping in delta_x if needed
                # (Simple subtraction is often okay for small deltas in iLQR)
                
                delta_u = alpha * ds[k] + Ks[k] @ delta_x
                
                u_applied = U_old[k] + delta_u
                
                # Box Constraints (Clamping) - Matches C++ usage of umax
                u_applied = np.clip(u_applied, -self.planner_settings.umax, self.planner_settings.umax)
                
                U_new[k] = u_applied
                
                # Integration
                try:
                    X_new[k+1] = self._rk4_step(x_k, u_applied, dt, os_list[k], os_list[k+1])
                except (ValueError, ArithmeticError):
                    diverged = True
                    break
            
            if diverged:
                continue
                
            cost = self._cost_function(X_new, U_new, goals)
            
            if cost < current_cost:
                return X_new, U_new, cost, alpha
        
        # Failure to improve
        return X_old, U_old, current_cost, 0.0

    # ---------------------------------------------------------
    # Helpers: Dynamics, Costs, Initialization
    # ---------------------------------------------------------

    def _cost_function(self, X: np.ndarray, U: np.ndarray, goals: GoalList) -> float:
        """
        Calculates Total Cost. Mirrors OldPlanner::cost2Func.
        Currently implements Quadratic Cost.
        """
        # Note: In C++, goals are handled inside stepcost_vec/quat.
        # Here we assume a regulation to zero for simplicity or implement goal diff.
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

    def _trajOptBefore(self, x0: np.ndarray, dt: float, N: int, os_list: List[Orbital_State]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialization Phase. Mirrors OldPlanner::trajOptBefore.
        Generates initial guess (currently Zero Control / Damping).
        """
        # C++ supports B-Dot initialization. 
        # For this scratch rewrite, we start with a zero-control rollout 
        # which is a standard "Cold Start".
        
        U_init = np.zeros((N-1, self.nu))
        X_init = self._rollout(x0, U_init, dt, os_list)
        
        return X_init, U_init

    def _rollout(self, x0: np.ndarray, U: np.ndarray, dt: float, os_list: List[Orbital_State]) -> np.ndarray:
        N = U.shape[0] + 1
        X = np.zeros((N, self.nx))
        X[0] = x0
        
        for k in range(N-1):
            X[k+1] = self._rk4_step(X[k], U[k], dt, os_list[k], os_list[k+1])
        return X

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float, os_prev: Orbital_State, os_next: Orbital_State) -> np.ndarray:
        """
        RK4 Integration. Matches Satellite.dynamics_for_solver.
        """
        # Call dynamics at t=0, t=dt/2, t=dt
        k1 = self.est_sat.dynamics_for_solver(0.0, x, u, os_prev, os_next)
        k2 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k1, u, os_prev, os_next)
        k3 = self.est_sat.dynamics_for_solver(dt/2.0, x + 0.5*dt*k2, u, os_prev, os_next)
        k4 = self.est_sat.dynamics_for_solver(dt, x + dt*k3, u, os_prev, os_next)
        
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize Quaternion (indices 3:7)
        # Using fast manual normalization to avoid import overhead
        q = x_next[3:7]
        nm = np.dot(q, q)
        if nm > 1e-12:
            x_next[3:7] /= np.sqrt(nm)
            
        return x_next

    def _linearize_dynamics_local(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local implementation of linearization to ensure B-matrix shape correctness.
        Replaces direct call to sat.dynJacCore to fix the (Control x State) issue.
        """
        sat = self.est_sat
        
        # Extract Env
        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        # Extract State
        w = x[0:3]
        q = x[3:7]
        RWhs = x[7:]
        
        invJ_noRW = sat.invJ_noRW
        J = sat.J_0

        # Rotate Env to Body
        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B @ R
        B_B = rmat_ECI2B @ B
        S_B = rmat_ECI2B @ S
        V_B = rmat_ECI2B @ V

        # Derivatives of Rotation
        dR_B__dq = drotmatTvecdq(q, R)
        dB_B__dq = drotmatTvecdq(q, B)
        dV_B__dq = drotmatTvecdq(q, V)
        dS_B__dq = drotmatTvecdq(q, S)
        
        vecs = {
            "b": B_B, "r": R_B, "s": S_B, "v": V_B, "rho": rho,
            "db": dB_B__dq, "ds": dS_B__dq, "dv": dV_B__dq, "dr": dR_B__dq,
            "os": orbital_state
        }

        # --- A Matrix (dxdot/dx) ---
        ddist_torq__dx, _ = sat.dist_torques_jacobian(x, vecs)
        
        # Note: calling actuator methods with (u, x, vecs) NOT (u, sat, x, vecs)
        dact_torq__dbase = sum([sat.actuators[j].dtorq__dbasestate(u[j], x, orbital_state) for j in range(len(sat.actuators))], np.zeros((7, 3)))
        
        dxdot__dx = np.zeros((sat.state_len, sat.state_len))
        
        # Kinematics
        dxdot__dx[3, 4:7] = 0.5 * w
        dxdot__dx[4:7, 3] = -0.5 * w
        dxdot__dx[4:7, 4:7] = 0.5 * skewsym(w)
        dxdot__dx[0:3, 3:7] = 0.5 * Wmat(q).T
        
        dxdot__dx[:, 0:3] += ddist_torq__dx @ invJ_noRW
        dxdot__dx[0:7, 0:3] += dact_torq__dbase @ invJ_noRW
        dxdot__dx[0:3, 0:3] += (-skewsym(w @ J) + J @ skewsym(w)) @ invJ_noRW

        # --- B Matrix (dxdot/du) ---
        # NOTE: This variable name 'dxdot__du' in Satellite.py usually implies B^T shape (Control, State).
        # We construct it here as B^T (Control, State) to match logic, then transpose at return.
        
        # 1. Base Actuators
        # dact_torq__du is stacked (N_act, 3).
        dact_torq__du = np.vstack([sat.actuators[j].dtorq__du(u[j], x, orbital_state) for j in range(len(sat.actuators))])
        
        # (Control, State) shape
        dxdot__du_T = np.zeros((sat.control_len, sat.state_len))
        
        # Fill angular acceleration part (Cols 0:3)
        # (N_act, 3) @ (3, 3) -> (N_act, 3)
        dxdot__du_T[:, 0:3] = dact_torq__du @ invJ_noRW

        # 2. Reaction Wheels
        if sat.number_RW > 0:
            dact_torq__dh = np.vstack([sat.actuators[j].dtorq__dh(u[j], x,orbital_state) for j in range(len(sat.actuators))])
            RWjs = np.array([sat.actuators[j].J for j in sat.momentum_inds])
            RWaxes = np.vstack([sat.actuators[j].axis for j in sat.momentum_inds])
            mRWjs = np.diagflat(RWjs)
            
            # A Matrix updates
            dxdot__dx[0:3, 0:3] += -skewsym(RWhs @ RWaxes) @ invJ_noRW
            dxdot__dx[7:, 0:3] += (dact_torq__dh + np.cross(RWaxes, w)) @ invJ_noRW
            dxdot__dx[0:7, 7:] = np.hstack([sat.actuators[j].dstor_torq__dbasestate(u[j], x, orbital_state) for j in range(len(sat.actuators))])
            dxdot__dx[7:, 7:] = np.diagflat([sat.actuators[j].dstor_torq__dh(u[j], x, orbital_state) for j in sat.momentum_inds])
            
            # Coupling term A adjustment
            dxdot__dx[:, 7:] -= dxdot__dx[:, 0:3] @ RWaxes.T @ mRWjs
            
            # B Matrix updates (RW direct term)
            # This fills the bottom right block of B^T (Controls vs RW states)
            # In Satellite.py: dxdot__du[:, 7:] = block_diag...
            
            # We iterate to robustly fill the block diagonal
            for i, act in enumerate(sat.actuators):
                if isinstance(act, type(sat.rw_actuators[0])):
                    rw_idx = np.where(sat.momentum_inds == i)[0][0]
                    # Map control 'i' to state '7+rw_idx'
                    val = act.dstor_torq__du(u[i], x, orbital_state)
                    dxdot__du_T[i, 7+rw_idx] = val
            
            # Coupling term B adjustment
            # Original: dxdot__du[:, 7:] -= dxdot__du[:, 0:3] @ RWaxes.T @ mRWjs
            dxdot__du_T[:, 7:] -= dxdot__du_T[:, 0:3] @ RWaxes.T @ mRWjs
            
        return dxdot__dx, dxdot__du_T

    def _propagate_environment(self, os_0: Orbital_State, t_start: float, t_end: float, dt: float, N: int) -> List[Orbital_State]:
        """
        Propagates orbit and creates lightweight Orbital_States.
        """
        orb = Orbit(os0=os_0, end_time=t_end + 10*dt*TimeConstants.sec2cent, dt=dt, fast=True)
        tp_orbit = orb.get_range(t_start, t_end, dt)
        vecs_raw = tp_orbit.get_vecs() 
        times = np.array(tp_orbit.times)
        
        # Safe Slicing
        R = np.array(vecs_raw[0])[:N]
        V = np.array(vecs_raw[1])[:N]
        B = np.array(vecs_raw[2])[:N]
        S = np.array(vecs_raw[3])[:N]
        times = times[:N]
        
        # Padding
        if len(R) < N:
            pad = N - len(R)
            R = np.pad(R, ((0,pad), (0,0)), 'edge')
            V = np.pad(V, ((0,pad), (0,0)), 'edge')
            B = np.pad(B, ((0,pad), (0,0)), 'edge')
            S = np.pad(S, ((0,pad), (0,0)), 'edge')
            times = np.pad(times, (0, pad), 'edge')

        ephem = os_0.ephem
        os_list = []
        for i in range(N):
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
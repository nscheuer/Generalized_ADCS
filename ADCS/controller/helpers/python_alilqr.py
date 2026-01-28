"""
Python ALILQR (Augmented Lagrangian Iterative Linear Quadratic Regulator)

A Python implementation of the trajectory optimization loop that uses C++ 
subroutines for performance-critical operations but runs the outer iteration
logic in Python. This enables:

- Step-by-step inspection of trajectories at each iteration
- Visualization of convergence between inner/outer loops
- Analysis of constraint handling and penalty evolution
- Easy debugging and modification of the algorithm flow

The C++ planner (tplaunch.Planner) provides:
- generateInitialTrajectory: Roll out trajectory from controls
- backwardPass: Compute feedback gains via Riccati-like equations
- forwardPass: Line-search-based trajectory update
- ilqrStep: Combined backward+forward pass (optional)
- maxViol: Compute constraint violations
- incrementAugLag: Update Lagrange multipliers and penalties
- ilqrBreak / outerBreak: Convergence checks
- cost2Func: Augmented Lagrangian cost evaluation

Usage:
    from ADCS.controller.helpers.python_alilqr import PythonALILQR
    
    py_alilqr = PythonALILQR(planner, debug_callback=my_callback)
    result = py_alilqr.optimize(dt, initial_traj, vecs, cost_settings, alilqr_settings)
"""
from __future__ import annotations

__all__ = ["PythonALILQR", "IterationData", "OptimizationResult"]

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Any, Dict
from numpy.typing import NDArray
import copy


@dataclass
class IterationData:
    """Container for data collected at each iteration."""
    outer_iter: int
    inner_iter: int
    
    # Trajectory
    Xset: NDArray[np.float64]
    Uset: NDArray[np.float64]
    TQset: Optional[NDArray[np.float64]] = None
    
    # Costs
    LA: float = 0.0  # Augmented Lagrangian cost
    LA_nc: float = 0.0  # Cost without constraints
    dLA: float = 0.0  # Change in cost
    
    # Constraint info
    cmax: float = 0.0  # Max constraint violation
    clist: Optional[NDArray[np.float64]] = None  # Per-constraint violations
    
    # Convergence metrics
    grad: float = 0.0  # Gradient norm proxy
    
    # Regularization
    rho: float = 0.0
    drho: float = 0.0
    
    # Augmented Lagrangian
    mu: float = 0.0  # Penalty parameter
    lambdaSet: Optional[NDArray[np.float64]] = None
    muSet: Optional[NDArray[np.float64]] = None
    
    # Backward pass results
    Kset: Optional[NDArray[np.float64]] = None
    dset: Optional[NDArray[np.float64]] = None
    delV: Optional[NDArray[np.float64]] = None
    
    # Flags
    ls_converged: bool = True
    break_reason: str = ""


@dataclass
class OptimizationResult:
    """Final result of the optimization."""
    success: bool
    Xset: NDArray[np.float64]
    Uset: NDArray[np.float64]
    TQset: NDArray[np.float64]
    Kset: NDArray[np.float64]
    times: NDArray[np.float64]
    
    # Final metrics
    final_cost: float
    final_cmax: float
    final_grad: float
    
    # Iteration history
    iterations: List[IterationData] = field(default_factory=list)
    
    # Statistics
    total_outer_iters: int = 0
    total_inner_iters: int = 0
    final_mu: float = 0.0


class PythonALILQR:
    """
    Python-driven ALILQR optimizer using C++ subroutines.
    
    This class provides the same optimization as the C++ alilqr() method,
    but with the iteration control in Python for easier debugging and analysis.
    
    The algorithm structure is:
        OUTER LOOP (Augmented Lagrangian):
            - Updates penalty (mu) and Lagrange multipliers (lambda)
            - Runs until constraints satisfied or max iterations
            
            INNER LOOP (iLQR):
                - BACKWARD PASS: Compute gains K and feedforward d
                - FORWARD PASS: Line search to update trajectory
                - Check convergence (gradient, cost change)
    
    Parameters
    ----------
    planner : tplaunch.Planner
        The C++ planner instance with bound methods
    debug_callback : callable, optional
        Function called after each iteration: callback(iteration_data)
    verbose : bool
        Print progress information
    """
    
    def __init__(
        self,
        planner,  # tplaunch.Planner
        debug_callback: Optional[Callable[[IterationData], None]] = None,
        verbose: bool = False
    ):
        self.planner = planner
        self.debug_callback = debug_callback
        self.verbose = verbose
        
    def optimize(
        self,
        dt: float,
        initial_traj: Tuple[NDArray, NDArray, NDArray, NDArray],
        vecs: Tuple,
        cost_settings: Tuple,
        alilqr_settings: Tuple,
        is_first_search: bool = True,
        collect_all: bool = True
    ) -> OptimizationResult:
        """
        Run the full ALILQR optimization.
        
        Parameters
        ----------
        dt : float
            Time step for dynamics integration
        initial_traj : tuple
            (Xset, Uset, times, TQset) - Initial trajectory guess
        vecs : tuple
            Environment vectors (t, R, V, B, S, A, E, p, rho)
        cost_settings : tuple
            Cost function weights and settings
        alilqr_settings : tuple
            (line_search_settings, auglag_settings, break_settings, reg_settings)
        is_first_search : bool
            Whether this is the first (exploration) pass
        collect_all : bool
            Store all iteration data (memory intensive for long runs)
            
        Returns
        -------
        OptimizationResult
            Optimization result with trajectory and history
        """
        # Unpack settings
        line_search_settings, auglag_settings, break_settings, reg_settings = alilqr_settings
        
        # Unpack auglag settings
        lam_init, lam_max, mu_init, mu_max, mu_scale = auglag_settings
        
        # Unpack break settings (need to handle xmax as array)
        (max_outer_iter, max_inner_iter, max_total_iter, 
         grad_tol, ilqr_cost_tol, cost_tol, z_count_lim, 
         cmax_target, max_cost, xmax) = break_settings
        
        # Unpack reg settings
        (reg_init, reg_min, reg_max, reg_scale, reg_bump,
         reg_min_cond, rand_ratio, use_dyn_hess, use_constr_hess) = reg_settings
        
        # Unpack initial trajectory
        Xset, Uset, times, TQset = initial_traj
        N = Xset.shape[1]
        
        # Get constraint count from planner (via maxViol output shape)
        # Initialize augmented Lagrangian variables
        constraint_N = self._get_constraint_count(initial_traj, vecs)
        
        lambdaSet = np.ones((constraint_N, N)) * lam_init
        muSet = np.ones((constraint_N, N)) * (mu_init / mu_scale)
        mu = mu_init / mu_scale
        
        auglag_vals = (lambdaSet, mu, muSet)
        
        # Regenerate initial trajectory to ensure consistency
        traj = self.planner.generateInitialTrajectory(dt, Xset[:, 0].copy(), Uset.copy(), vecs)
        Xset, Uset, times, TQset = traj
        
        # Get initial constraint violations and increment auglag
        clist, cmax = self.planner.maxViol(traj, vecs, auglag_vals)
        auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
        lambdaSet, mu, muSet = auglag_vals
        
        # Zero auglag for clean cost calculation
        auglag_vals_clean = (np.zeros_like(lambdaSet), 0.0, np.zeros_like(muSet))
        
        # Initial costs
        LA = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
        LA_nc = self.planner.cost2Func(traj, vecs, auglag_vals_clean, cost_settings)
        
        # Initialize tracking
        iterations: List[IterationData] = []
        total_iter = 0
        regs = (reg_init, 0.0)  # (rho, drho)
        
        grad = 1e10
        dLA = 0.0
        dla_z_count = 0
        
        # Store last backward pass results for final Kset
        last_Kset = None
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Python ALILQR: N={N}, dt={dt:.3f}")
            print(f"Initial cost: LA={LA:.6e}, LA_nc={LA_nc:.6e}, cmax={cmax:.6e}")
            print(f"{'='*70}")
        
        # =====================================================================
        # OUTER LOOP: Augmented Lagrangian
        # =====================================================================
        outer_break_reason = ""
        
        for j in range(max_outer_iter):
            if self.verbose:
                print(f"\n--- Outer iteration {j} ---")
                print(f"    mu={mu:.2e}, max(lambda)={np.max(np.abs(lambdaSet)):.2e}")
            
            # Reset for inner loop
            cmax = 0.0
            dla_z_count = 0
            clist = np.zeros((constraint_N, N))
            dLA = 0.0
            
            # Reset regularization at start of each outer iteration
            regs = (reg_init, 0.0)
            
            # Compute cost at start of inner loop
            LA = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
            
            # =================================================================
            # INNER LOOP: iLQR
            # =================================================================
            inner_break_reason = ""
            
            for ii in range(max_inner_iter):
                total_iter += 1
                use_dist = not is_first_search
                
                # -------------------------------------------------------------
                # BACKWARD PASS
                # -------------------------------------------------------------
                bp_results, regs = self.planner.backwardPass(
                    dt, traj, vecs, auglag_vals, regs,
                    cost_settings, reg_settings, use_dist
                )
                Kset_bp, dset, delV = bp_results
                last_Kset = Kset_bp
                
                # -------------------------------------------------------------
                # FORWARD PASS (with line search)
                # -------------------------------------------------------------
                traj_new, newLA, regs = self.planner.forwardPass(
                    dt, traj, vecs, auglag_vals, bp_results, regs,
                    cost_settings, reg_settings, line_search_settings, use_dist
                )
                
                Xset_new, Uset_new, times_new, TQset_new = traj_new
                
                # Compute gradient proxy
                if dset.shape[1] > 0 and Uset_new.shape[1] > 0:
                    cols = min(dset.shape[1], Uset_new.shape[1])
                    grad = np.mean(
                        np.max(np.abs(dset[:, :cols]) / (np.abs(Uset_new[:, :cols]) + 1), axis=0)
                    )
                else:
                    grad = 0.0
                
                # Get constraint violations
                clist, cmax = self.planner.maxViol(traj_new, vecs, auglag_vals)
                
                # Compute cost change
                dLA = abs(newLA - LA)
                dla_z_count += 1
                if dLA != 0:
                    dla_z_count = 0
                
                # Update trajectory and cost
                LA_old = LA
                LA = newLA
                LA_nc = self.planner.cost2Func(traj_new, vecs, auglag_vals_clean, cost_settings)
                traj = traj_new
                Xset, Uset, times, TQset = traj
                
                rho, drho = regs
                
                # Collect iteration data
                iter_data = IterationData(
                    outer_iter=j,
                    inner_iter=ii,
                    Xset=Xset.copy(),
                    Uset=Uset.copy(),
                    TQset=TQset.copy() if TQset is not None else None,
                    LA=LA,
                    LA_nc=LA_nc,
                    dLA=dLA,
                    cmax=cmax,
                    clist=clist.copy(),
                    grad=grad,
                    rho=rho,
                    drho=drho,
                    mu=mu,
                    lambdaSet=lambdaSet.copy(),
                    muSet=muSet.copy(),
                    Kset=Kset_bp.copy() if isinstance(Kset_bp, np.ndarray) else None,
                    dset=dset.copy(),
                    delV=delV.copy() if delV is not None else None,
                    ls_converged=(LA <= LA_old),
                    break_reason=""
                )
                
                if collect_all:
                    iterations.append(iter_data)
                
                # Call debug callback
                if self.debug_callback:
                    self.debug_callback(iter_data)
                
                if self.verbose:
                    print(f"    inner {ii}: LA={LA:.4e}, dLA={dLA:.2e}, "
                          f"grad={grad:.2e}, cmax={cmax:.2e}, rho={rho:.2e}")
                
                # Check inner break condition
                if self.planner.ilqrBreak(
                    grad, LA, dLA, dla_z_count, cmax, total_iter,
                    break_settings, j, ii, False
                ):
                    inner_break_reason = f"ilqrBreak at ii={ii}"
                    if self.verbose:
                        print(f"    INNER BREAK: {inner_break_reason}")
                    break
            
            # =================================================================
            # End of inner loop
            # =================================================================
            
            # Check outer break conditions
            outer_break_cond = self.planner.outerBreak(
                auglag_vals, cmax, break_settings, auglag_settings, j
            )
            ilqr_break_for_outer = self.planner.ilqrBreak(
                grad, LA, dLA, dla_z_count, cmax, total_iter,
                break_settings, j, ii, True
            )
            
            if outer_break_cond and j > 2 and ilqr_break_for_outer:
                outer_break_reason = f"outerBreak at j={j}, cmax={cmax:.2e}"
                if self.verbose:
                    print(f"OUTER BREAK: {outer_break_reason}")
                break
            
            # Update augmented Lagrangian parameters
            auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
            lambdaSet, mu, muSet = auglag_vals
        
        # =====================================================================
        # Build result
        # =====================================================================
        
        # Package Kset - handle 3D cube from backward pass
        if last_Kset is not None:
            if len(last_Kset.shape) == 3:
                # Cube: (ctrl_dim, reduced_state_dim, N-1) -> package to (ctrl_dim*state_dim, N-1)
                ctrl_dim, state_dim, T = last_Kset.shape
                Kset_final = last_Kset.reshape((ctrl_dim * state_dim, T), order='C')
            else:
                Kset_final = last_Kset
        else:
            Kset_final = np.zeros((1, N-1))
        
        result = OptimizationResult(
            success=(cmax < cmax_target) or (mu >= mu_max),
            Xset=Xset,
            Uset=Uset,
            TQset=TQset if TQset is not None else np.zeros((3, N)),
            Kset=Kset_final,
            times=times,
            final_cost=LA,
            final_cmax=cmax,
            final_grad=grad,
            iterations=iterations,
            total_outer_iters=j + 1,
            total_inner_iters=total_iter,
            final_mu=mu
        )
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Optimization complete: outer={j+1}, total_inner={total_iter}")
            print(f"Final: cost={LA:.6e}, cmax={cmax:.6e}, grad={grad:.6e}")
            print(f"{'='*70}")
        
        return result
    
    def _get_constraint_count(
        self,
        traj: Tuple[NDArray, NDArray, NDArray, NDArray],
        vecs: Tuple
    ) -> int:
        """Infer constraint count from maxViol output."""
        # Create dummy auglag with mu=0 to get clist shape
        Xset = traj[0]
        N = Xset.shape[1]
        
        # Try with a small initial guess
        test_auglag = (np.zeros((1, N)), 0.0, np.zeros((1, N)))
        try:
            clist, _ = self.planner.maxViol(traj, vecs, test_auglag)
            return clist.shape[0]
        except:
            # Fallback - common values for spacecraft
            return 6  # 3 control constraints + 3 rate constraints
    
    def optimize_step_by_step(
        self,
        dt: float,
        initial_traj: Tuple[NDArray, NDArray, NDArray, NDArray],
        vecs: Tuple,
        cost_settings: Tuple,
        alilqr_settings: Tuple,
        is_first_search: bool = True
    ):
        """
        Generator version for step-by-step iteration.
        
        Yields IterationData after each inner iteration, allowing external 
        control of the loop and real-time visualization.
        
        Usage:
            for iter_data in py_alilqr.optimize_step_by_step(...):
                plot_trajectory(iter_data.Xset, iter_data.Uset)
                if some_condition:
                    break  # Can stop early
        """
        # Unpack settings
        line_search_settings, auglag_settings, break_settings, reg_settings = alilqr_settings
        
        lam_init, lam_max, mu_init, mu_max, mu_scale = auglag_settings
        
        (max_outer_iter, max_inner_iter, max_total_iter, 
         grad_tol, ilqr_cost_tol, cost_tol, z_count_lim, 
         cmax_target, max_cost, xmax) = break_settings
        
        (reg_init, reg_min, reg_max, reg_scale, reg_bump,
         reg_min_cond, rand_ratio, use_dyn_hess, use_constr_hess) = reg_settings
        
        Xset, Uset, times, TQset = initial_traj
        N = Xset.shape[1]
        
        constraint_N = self._get_constraint_count(initial_traj, vecs)
        
        lambdaSet = np.ones((constraint_N, N)) * lam_init
        muSet = np.ones((constraint_N, N)) * (mu_init / mu_scale)
        mu = mu_init / mu_scale
        
        auglag_vals = (lambdaSet, mu, muSet)
        
        traj = self.planner.generateInitialTrajectory(dt, Xset[:, 0].copy(), Uset.copy(), vecs)
        Xset, Uset, times, TQset = traj
        
        clist, cmax = self.planner.maxViol(traj, vecs, auglag_vals)
        auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
        lambdaSet, mu, muSet = auglag_vals
        
        auglag_vals_clean = (np.zeros_like(lambdaSet), 0.0, np.zeros_like(muSet))
        
        LA = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
        
        total_iter = 0
        regs = (reg_init, 0.0)
        grad = 1e10
        dLA = 0.0
        dla_z_count = 0
        
        for j in range(max_outer_iter):
            cmax = 0.0
            dla_z_count = 0
            clist = np.zeros((constraint_N, N))
            dLA = 0.0
            regs = (reg_init, 0.0)
            
            LA = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
            
            for ii in range(max_inner_iter):
                total_iter += 1
                use_dist = not is_first_search
                
                bp_results, regs = self.planner.backwardPass(
                    dt, traj, vecs, auglag_vals, regs,
                    cost_settings, reg_settings, use_dist
                )
                Kset_bp, dset, delV = bp_results
                
                traj_new, newLA, regs = self.planner.forwardPass(
                    dt, traj, vecs, auglag_vals, bp_results, regs,
                    cost_settings, reg_settings, line_search_settings, use_dist
                )
                
                Xset_new, Uset_new, times_new, TQset_new = traj_new
                
                if dset.shape[1] > 0 and Uset_new.shape[1] > 0:
                    cols = min(dset.shape[1], Uset_new.shape[1])
                    grad = np.mean(
                        np.max(np.abs(dset[:, :cols]) / (np.abs(Uset_new[:, :cols]) + 1), axis=0)
                    )
                else:
                    grad = 0.0
                
                clist, cmax = self.planner.maxViol(traj_new, vecs, auglag_vals)
                
                dLA = abs(newLA - LA)
                dla_z_count += 1
                if dLA != 0:
                    dla_z_count = 0
                
                LA_old = LA
                LA = newLA
                LA_nc = self.planner.cost2Func(traj_new, vecs, auglag_vals_clean, cost_settings)
                traj = traj_new
                Xset, Uset, times, TQset = traj
                
                rho, drho = regs
                
                iter_data = IterationData(
                    outer_iter=j,
                    inner_iter=ii,
                    Xset=Xset.copy(),
                    Uset=Uset.copy(),
                    TQset=TQset.copy() if TQset is not None else None,
                    LA=LA,
                    LA_nc=LA_nc,
                    dLA=dLA,
                    cmax=cmax,
                    clist=clist.copy(),
                    grad=grad,
                    rho=rho,
                    drho=drho,
                    mu=mu,
                    lambdaSet=lambdaSet.copy(),
                    muSet=muSet.copy(),
                    Kset=Kset_bp.copy() if isinstance(Kset_bp, np.ndarray) else None,
                    dset=dset.copy(),
                    delV=delV.copy() if delV is not None else None,
                    ls_converged=(LA <= LA_old),
                    break_reason=""
                )
                
                yield iter_data
                
                if self.planner.ilqrBreak(
                    grad, LA, dLA, dla_z_count, cmax, total_iter,
                    break_settings, j, ii, False
                ):
                    break
            
            outer_break_cond = self.planner.outerBreak(
                auglag_vals, cmax, break_settings, auglag_settings, j
            )
            ilqr_break_for_outer = self.planner.ilqrBreak(
                grad, LA, dLA, dla_z_count, cmax, total_iter,
                break_settings, j, ii, True
            )
            
            if outer_break_cond and j > 2 and ilqr_break_for_outer:
                break
            
            auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
            lambdaSet, mu, muSet = auglag_vals


def run_with_visualization(
    py_alilqr: PythonALILQR,
    dt: float,
    initial_traj: Tuple[NDArray, NDArray, NDArray, NDArray],
    vecs: Tuple,
    cost_settings: Tuple,
    alilqr_settings: Tuple,
    is_first_search: bool = True,
    plot_every: int = 1,
    plot_fn: Optional[Callable[[IterationData], None]] = None
) -> OptimizationResult:
    """
    Run optimization with optional plotting at regular intervals.
    
    Parameters
    ----------
    py_alilqr : PythonALILQR
        The optimizer instance
    dt : float
        Time step
    initial_traj : tuple
        Initial trajectory
    vecs : tuple
        Environment vectors
    cost_settings : tuple
        Cost settings
    alilqr_settings : tuple
        ALILQR settings
    is_first_search : bool
        First search flag
    plot_every : int
        Plot every N iterations
    plot_fn : callable
        Custom plotting function (iter_data -> None)
        
    Returns
    -------
    OptimizationResult
    """
    iterations = []
    
    def callback(iter_data):
        iterations.append(iter_data)
        total = len(iterations)
        if total % plot_every == 0:
            if plot_fn:
                plot_fn(iter_data)
            else:
                print(f"Iter {total}: outer={iter_data.outer_iter}, inner={iter_data.inner_iter}, "
                      f"cost={iter_data.LA:.4e}, cmax={iter_data.cmax:.4e}")
    
    # Run with callback
    old_callback = py_alilqr.debug_callback
    py_alilqr.debug_callback = callback
    
    result = py_alilqr.optimize(
        dt, initial_traj, vecs, cost_settings, alilqr_settings, is_first_search
    )
    
    py_alilqr.debug_callback = old_callback
    result.iterations = iterations
    
    return result

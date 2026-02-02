"""
Python ALILQR wrapper V2 - Follows C++ implementation EXACTLY.

This implementation uses C++ bindings for all core computations but pulls out
to Python at each iteration for visualization and debugging.

The goal is to match the C++ `alilqr()` function step-by-step:
1. Initialize lambdaSet, muSet, mu
2. Call generateInitialTrajectory
3. Get initial constraint violations and increment auglag
4. Outer loop (j = 0 to maxOuterIter):
   a. Reset cmaxtmp, dlaZcount, clist, dLA, regs
   b. Compute initial cost
   c. Inner loop (ii = 0 to maxIlqrIter):
      - Call ilqrStep (backward pass + forward pass with line search)
      - Update dLA, dlaZcount, LA, traj
      - Check ilqrBreak
   d. Check outerBreak
   e. incrementAugLag
5. Package results
"""
from __future__ import annotations

__all__ = ["PythonALILQRv2", "OptimizationResult", "IterationData"]

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Any
from numpy.typing import NDArray


@dataclass
class IterationData:
    """Data collected at each inner iteration."""
    outer_iter: int
    inner_iter: int
    total_iter: int
    
    # Trajectory
    Xset: NDArray[np.float64]
    Uset: NDArray[np.float64]
    TQset: Optional[NDArray[np.float64]]
    times: NDArray[np.float64]
    
    # Costs
    LA: float          # Augmented Lagrangian cost
    LA_nc: float       # Cost without constraints
    dLA: float         # Change in cost
    
    # Constraints
    cmax: float        # Max constraint violation
    clist: NDArray[np.float64]  # All constraint violations
    
    # Convergence
    grad: float        # Gradient norm proxy
    rho: float         # Regularization
    drho: float        # Regularization change
    
    # Augmented Lagrangian
    mu: float
    lambdaSet: NDArray[np.float64]
    muSet: NDArray[np.float64]
    
    # Gains (from backward pass)
    Kset: Optional[NDArray[np.float64]] = None
    
    # Labels
    pass_label: str = ""
    break_reason: str = ""


@dataclass  
class OptimizationResult:
    """Result from optimization."""
    Xset: NDArray[np.float64]
    Uset: NDArray[np.float64]
    TQset: NDArray[np.float64]
    times: NDArray[np.float64]
    Kset: NDArray[np.float64]
    
    final_cost: float
    final_cmax: float
    final_grad: float
    final_mu: float
    
    total_outer_iters: int
    total_inner_iters: int
    
    iterations: List[IterationData] = field(default_factory=list)
    break_reason: str = ""


class PythonALILQRv2:
    """
    Python ALILQR wrapper that exactly matches C++ implementation.
    
    Uses C++ bindings for computations but exposes each iteration to Python.
    """
    
    def __init__(
        self,
        planner,
        debug_callback: Optional[Callable[[IterationData], None]] = None,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        planner : tplaunch.Planner
            C++ planner object with Python bindings
        debug_callback : callable, optional
            Function called at each iteration with IterationData
        verbose : bool
            Print progress to stdout
        """
        self.planner = planner
        self.debug_callback = debug_callback
        self.verbose = verbose
    
    def set_callback(self, callback: Callable[[IterationData], None]) -> None:
        """Set the debug callback."""
        self.debug_callback = callback
    
    def _scale_rw_controls(self, Uset: NDArray) -> NDArray:
        """
        Scale RW/magic controls from optimizer units to physical units.
        
        The optimizer uses scaled controls for better conditioning:
            u_scaled = u_physical / NONMTQ_TORQ_SCALE
        
        This converts back to physical units:
            u_physical = u_scaled * NONMTQ_TORQ_SCALE
        """
        # Get scaling factor and actuator counts from C++ planner
        scale = self.planner.get_nonmtq_torq_scale()
        if scale == 1.0:
            return Uset  # No scaling needed
        
        n_mtq = self.planner.get_number_MTQ()
        n_rw = self.planner.get_number_RW()
        n_magic = self.planner.get_number_magic()
        
        if n_rw + n_magic == 0:
            return Uset  # No RW or magic actuators
        
        # Scale RW and magic rows (indices n_mtq onwards)
        Uset_scaled = Uset.copy()
        rw_magic_start = n_mtq
        rw_magic_end = n_mtq + n_rw + n_magic
        Uset_scaled[rw_magic_start:rw_magic_end, :] *= scale
        
        return Uset_scaled
    
    def optimize(
        self,
        dt: float,
        initial_traj: Tuple,  # (Xset, Uset, times, TQset)
        vecs: Tuple,          # Environment vectors
        cost_settings: Tuple,
        alilqr_settings: Tuple,
        is_first_search: bool = True,
        collect_all: bool = True,
        pass_label: str = ""
    ) -> OptimizationResult:
        """
        Run ALILQR optimization - follows C++ alilqr() EXACTLY.
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        initial_traj : tuple
            (Xset, Uset, times, TQset) from prepareForAlilqr
        vecs : tuple
            Environment vectors from prepareForAlilqr
        cost_settings : tuple
            Cost function settings
        alilqr_settings : tuple
            (line_search_settings, auglag_settings, break_settings, reg_settings)
        is_first_search : bool
            If True, don't use disturbances in dynamics
        collect_all : bool
            If True, store all iteration data
        pass_label : str
            Label for this optimization pass (e.g., "Pass1", "Pass2")
            
        Returns
        -------
        OptimizationResult
        """
        # =====================================================================
        # UNPACK SETTINGS (matches C++ alilqr lines 1327-1343)
        # =====================================================================
        line_search_settings, auglag_settings, break_settings, reg_settings = alilqr_settings
        
        lam_init = auglag_settings[0]      # lagMultInit
        pen_init = auglag_settings[2]      # penInit  
        pen_scale = auglag_settings[4]     # penScale
        
        max_outer_iter = int(break_settings[0])
        max_inner_iter = int(break_settings[1])
        
        reg_init = reg_settings[0]
        
        # =====================================================================
        # UNPACK INITIAL TRAJECTORY (matches C++ lines 1345-1347)
        # =====================================================================
        Xset, Uset, times, TQset = initial_traj
        N = Xset.shape[1]
        constraint_N = self._get_constraint_count(initial_traj, vecs)
        
        # =====================================================================
        # INITIALIZE (matches C++ lines 1349-1357)
        # =====================================================================
        grad = 1.0 / 1e-10  # EPSVAR in C++
        total_iter = 0
        
        lambdaSet = np.ones((constraint_N, N)) * lam_init
        muSet = np.ones((constraint_N, N)) * (pen_init / pen_scale)
        mu = pen_init / pen_scale
        auglag_vals = (lambdaSet, mu, muSet)
        
        # =====================================================================
        # GENERATE INITIAL TRAJECTORY (matches C++ line 1359)
        # Note: C++ calls generateInitialTrajectory here, but this can cause NaN
        # with large timesteps. The traj from prepareForAlilqr is already valid.
        # =====================================================================
        traj = initial_traj  # Use directly - already generated by prepareForAlilqr
        
        # =====================================================================
        # GET INITIAL VIOLATIONS AND INCREMENT AUGLAG (matches C++ lines 1361-1363)
        # =====================================================================
        clist, cmax_init = self.planner.maxViol(traj, vecs, auglag_vals)
        auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
        lambdaSet, mu, muSet = auglag_vals
        
        # Clean auglag for computing cost without constraints
        auglag_vals_clean = (np.zeros_like(lambdaSet), 0.0, np.zeros_like(muSet))
        
        # =====================================================================
        # INITIAL COSTS (matches C++ lines 1364-1366)
        # =====================================================================
        LA0 = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
        LA = LA0
        LA_nc = self.planner.cost2Func(traj, vecs, auglag_vals_clean, cost_settings)
        
        # =====================================================================
        # MORE INITIALIZATION (matches C++ lines 1368-1375)
        # =====================================================================
        cmax = 0.0
        dla_z_count = 0
        dLA = 0.0
        newLA = LA
        regs = (reg_init, 0.0)  # (rho, drho)
        
        iterations: List[IterationData] = []
        last_Kset = None
        break_reason = ""
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Python ALILQR [{pass_label}]: N={N}, dt={dt:.3f}")
            print(f"Initial cost: LA={LA:.6e}, LA_nc={LA_nc:.6e}, cmax={cmax_init:.6e}")
            print(f"{'='*70}")
        
        # =====================================================================
        # OUTER LOOP (matches C++ line 1380: for j = 0 to maxOuterIter)
        # =====================================================================
        for j in range(max_outer_iter):
            if self.verbose:
                print(f"\n--- Outer iteration {j} ---")
                print(f"    mu={mu:.2e}, max(lambda)={np.max(np.abs(lambdaSet)):.2e}")
            
            # =================================================================
            # RESET FOR OUTER ITERATION (matches C++ lines 1383-1390)
            # =================================================================
            cmax = 0.0
            dla_z_count = 0
            clist = np.zeros((constraint_N, N))
            dLA = 0.0
            # stepsSinceRand = -1  # Not used in Python version
            
            regs = (reg_init, 0.0)  # Reset regularization
            
            # =================================================================
            # COMPUTE INITIAL COST (matches C++ line 1392)
            # =================================================================
            LA = self.planner.cost2Func(traj, vecs, auglag_vals, cost_settings)
            
            # =================================================================
            # INNER LOOP (matches C++ line 1397: for ii = 0 to maxIlqrIter)
            # =================================================================
            for ii in range(max_inner_iter):
                total_iter += 1
                
                # =============================================================
                # ILQR STEP (matches C++ line 1407)
                # This does: backward pass + forward pass with line search
                # =============================================================
                use_dist = not is_first_search
                
                ilqr_result = self.planner.ilqrStep(
                    dt, traj, vecs, auglag_vals, regs,
                    cost_settings, reg_settings, line_search_settings,
                    break_settings, use_dist
                )
                
                newLA, cmax, clist, grad, regs, traj = ilqr_result
                rho, drho = regs
                
                # =============================================================
                # UPDATE DLA AND COUNTER (matches C++ lines 1415-1421)
                # =============================================================
                dLA = abs(newLA - LA)
                dla_z_count += 1
                if dLA > 1e-10:  # Use tolerance for effectively zero
                    dla_z_count = 0
                
                # =============================================================
                # UPDATE LA AND TRAJECTORY (matches C++ lines 1424-1430)
                # =============================================================
                LA = newLA
                LA_nc = self.planner.cost2Func(traj, vecs, auglag_vals_clean, cost_settings)
                
                Xset, Uset, times, TQset = traj
                
                # =============================================================
                # COLLECT ITERATION DATA
                # =============================================================
                # Scale RW/magic controls to physical units for visualization
                Uset_physical = self._scale_rw_controls(Uset)
                
                iter_data = IterationData(
                    outer_iter=j,
                    inner_iter=ii,
                    total_iter=total_iter,
                    Xset=Xset.copy(),
                    Uset=Uset_physical,
                    TQset=TQset.copy() if TQset is not None else None,
                    times=times.copy(),
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
                    pass_label=pass_label,
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
                
                # =============================================================
                # CHECK INNER BREAK (matches C++ line 1433)
                # =============================================================
                if self.planner.ilqrBreak(
                    grad, LA, dLA, dla_z_count, cmax, total_iter,
                    break_settings, j, ii, False  # forOuter=False
                ):
                    if self.verbose:
                        print(f"    INNER BREAK at ii={ii}")
                    break
            
            # =================================================================
            # CHECK OUTER BREAK (matches C++ lines 1437-1440)
            # =================================================================
            outer_break_cond = self.planner.outerBreak(
                auglag_vals, cmax, break_settings, auglag_settings, j
            )
            ilqr_break_for_outer = self.planner.ilqrBreak(
                grad, LA, dLA, dla_z_count, cmax, total_iter,
                break_settings, j, ii, True  # forOuter=True
            )
            
            if outer_break_cond and j > 2 and ilqr_break_for_outer:
                break_reason = f"outerBreak at j={j}, cmax={cmax:.2e}"
                if self.verbose:
                    print(f"OUTER BREAK: {break_reason}")
                break
            
            # =================================================================
            # INCREMENT AUGLAG (matches C++ line 1443)
            # =================================================================
            auglag_vals = self.planner.incrementAugLag(auglag_vals, clist, auglag_settings)
            lambdaSet, mu, muSet = auglag_vals
        
        # =====================================================================
        # PACKAGE RESULTS (matches C++ lines 1445-1449)
        # =====================================================================
        Xset, Uset, times, TQset = traj
        
        # Scale RW/magic controls to physical units (matches C++ trajOptAfter)
        Uset_physical = self._scale_rw_controls(Uset)
        
        # Get final Kset by running one backward pass on the final trajectory
        # This matches C++ behavior: mat Kmat = packageK(get<0>(BPresults));
        try:
            use_dist = not is_first_search
            bp_results, _ = self.planner.backwardPass(
                dt, traj, vecs, auglag_vals, regs,
                cost_settings, reg_settings, use_dist
            )
            Kset_bp, _, _ = bp_results
            
            # Package Kset: C++ cube is (ctrl_dim, state_dim, N) but numpy receives as (N, ctrl_dim, state_dim)
            # We need to reshape to (ctrl_dim*state_dim, N) for storage and later use
            if isinstance(Kset_bp, np.ndarray) and Kset_bp.size > 0:
                if Kset_bp.ndim == 3:
                    # Numpy receives shape as (N, ctrl_dim, state_dim) due to how armadillo cubes are converted
                    T, ctrl_dim, state_dim = Kset_bp.shape
                    # Transpose to (ctrl_dim, state_dim, N) then reshape to (ctrl_dim * state_dim, N)
                    Kset_transposed = np.transpose(Kset_bp, (1, 2, 0))  # (ctrl_dim, state_dim, N)
                    Kset = Kset_transposed.reshape((ctrl_dim * state_dim, T), order='C')
                    if self.verbose:
                        print(f"Kset from backwardPass: {Kset_bp.shape} (N,ctrl,state) -> {Kset.shape}")
                elif Kset_bp.ndim == 2:
                    Kset = Kset_bp
                    if self.verbose:
                        print(f"Kset from backwardPass (2D): {Kset.shape}")
                else:
                    if self.verbose:
                        print(f"Warning: Unexpected Kset ndim={Kset_bp.ndim}")
                    Kset = np.zeros((Uset.shape[0] * (Xset.shape[0] - 1), N))
            else:
                if self.verbose:
                    print(f"Warning: Empty Kset from backwardPass")
                Kset = np.zeros((Uset.shape[0] * (Xset.shape[0] - 1), N))
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not compute Kset: {e}")
                import traceback
                traceback.print_exc()
            Kset = np.zeros((Uset.shape[0] * (Xset.shape[0] - 1), N))
        
        result = OptimizationResult(
            Xset=Xset,
            Uset=Uset_physical,
            TQset=TQset,
            times=times,
            Kset=Kset,
            final_cost=LA,
            final_cmax=cmax,
            final_grad=grad,
            final_mu=mu,
            total_outer_iters=j + 1,
            total_inner_iters=total_iter,
            iterations=iterations,
            break_reason=break_reason
        )
        
        return result
    
    def _get_constraint_count(self, traj: Tuple, vecs: Tuple) -> int:
        """Get the number of constraints from maxViol output."""
        Xset, Uset, times, TQset = traj
        N = Xset.shape[1]
        # Create dummy auglag to probe constraint count
        dummy_auglag = (np.zeros((1, N)), 0.0, np.zeros((1, N)))
        clist, _ = self.planner.maxViol(traj, vecs, dummy_auglag)
        return clist.shape[0]

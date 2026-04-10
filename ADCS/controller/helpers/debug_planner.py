"""
Debug wrapper for the trajectory planner that provides step-by-step output.

This module provides a DebugPlanner class that wraps the tplaunch.Planner
and exposes the same interface, while providing detailed diagnostic output
about the backward/forward passes, constraint violations, and active sets.

Usage:
    # In plan_and_track_exact.py, replace:
    #   self.planner = tplaunch.Planner(...)
    # With:
    #   from ADCS.controller.helpers.debug_planner import DebugPlanner
    #   self.planner = DebugPlanner(csat, ..., debug_level=2)
"""
from __future__ import annotations

__all__ = ["DebugPlanner"]

import numpy as np
from typing import Tuple, Optional, Dict, List, Any, TextIO
from numpy.typing import NDArray
from ADCS.controller.helpers.optional_dependencies import get_trajectory_planner_modules


class DebugPlanner:
    """
    A wrapper around tplaunch.Planner that provides debugging output.

    debug_level:
        0 = No debug output (passthrough to underlying planner)
        1 = Summary output (cost, max violation, iteration counts)
        2 = Detailed output (control patterns, constraint states)
        3 = Very detailed (per-timestep values)
    """

    def __init__(
        self,
        csat,
        systemSettings,
        alilqrSettings,
        alilqrSettings2,
        initialTrajSettings,
        costSettings,
        costSettings2,
        costSettings_tvlqr,
        debug_level: int = 1,
        log_file: Optional[str] = None
    ):
        tplaunch, _ = get_trajectory_planner_modules()
        # Store the underlying planner
        self._planner = tplaunch.Planner(
            csat, systemSettings, alilqrSettings, alilqrSettings2,
            initialTrajSettings, costSettings, costSettings2, costSettings_tvlqr
        )

        self.debug_level: int = debug_level
        self.log_file: Optional[str] = log_file
        self._log_handle: Optional[TextIO] = None

        # Extract control limits from alilqr settings for analysis
        # The settings structure: (lineSearch, auglag, break, reg)
        self._auglag_settings = alilqrSettings[1]  # (lam_init, lam_max, mu_init, mu_max, mu_scale)

        # Track iteration history
        self.iteration_history: List[Dict[str, Any]] = []
        self.control_history: List[NDArray[np.float64]] = []
        self.violation_history: List[float] = []

    def _log(self, msg: str, level: int = 1) -> None:
        """Log a message if debug_level is sufficient."""
        if self.debug_level >= level:
            print(msg)
            if self._log_handle:
                self._log_handle.write(msg + "\n")

    def _analyze_controls(self, Uset: NDArray[np.float64], u_limit: Optional[float] = None) -> Dict[str, List[Any]]:
        """Analyze control trajectory for oscillations and saturation."""
        if Uset.ndim == 1:
            Uset = Uset.reshape(-1, 1).T

        ctrl_dim, N = Uset.shape
        analysis = {
            'sign_changes': [],
            'saturation_count': [],
            'rapid_oscillations': [],
            'range': []
        }

        for ch in range(ctrl_dim):
            u_ch = Uset[ch, :]

            # Sign changes
            sign_changes = np.sum(np.diff(np.sign(u_ch)) != 0)
            analysis['sign_changes'].append(sign_changes)

            # Saturation
            if u_limit:
                at_sat = np.sum(np.abs(u_ch) > 0.95 * u_limit)
            else:
                at_sat = 0
            analysis['saturation_count'].append(at_sat)

            # Rapid oscillations (+-+ or -+- patterns)
            rapid_osc = 0
            if len(u_ch) > 2:
                signs = np.sign(u_ch)
                for i in range(len(signs) - 2):
                    if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
                        rapid_osc += 1
            analysis['rapid_oscillations'].append(rapid_osc)

            # Range
            analysis['range'].append((float(u_ch.min()), float(u_ch.max())))

        return analysis

    def _print_control_analysis(self, Uset: NDArray[np.float64], prefix: str = "") -> None:
        """Print control analysis summary."""
        analysis = self._analyze_controls(Uset)

        self._log(f"{prefix}Control Analysis:", 1)
        for ch in range(len(analysis['sign_changes'])):
            sc = analysis['sign_changes'][ch]
            ro = analysis['rapid_oscillations'][ch]
            r = analysis['range'][ch]
            warn = " [OSCILLATING!]" if ro > 5 else ""
            self._log(f"  Ch{ch}: sign_changes={sc}, rapid_osc={ro}, range=[{r[0]:.4e}, {r[1]:.4e}]{warn}", 1)

    # =========================================================================
    # Passthrough methods that match tplaunch.Planner interface
    # =========================================================================

    def readParameters(self):
        return self._planner.readParameters()

    def readDebug(self):
        return self._planner.readDebug()

    def updateParameters(self, *args):
        return self._planner.updateParameters(*args)

    def setVerbosity(self, verbosity: bool):
        return self._planner.setVerbosity(verbosity)

    def setquaternionTo3VecMode(self, val: int):
        return self._planner.setquaternionTo3VecMode(val)

    def getdt(self):
        return self._planner.getdt()

    def echo_int(self, x: int):
        return self._planner.echo_int(x)

    # =========================================================================
    # Main trajectory optimization with debug output
    # =========================================================================

    def trajOpt(self, vecsPy, N, time_start, time_end, x0, bdotOn):
        """
        Main trajectory optimization with debug output.
        """
        self._log("\n" + "=" * 70, 1)
        self._log("DEBUG PLANNER: trajOpt called", 1)
        self._log("=" * 70, 1)
        self._log(f"  N={N}, time_start={time_start:.6f}, time_end={time_end:.6f}", 1)
        self._log(f"  bdotOn={bdotOn}", 1)
        self._log(f"  x0 shape: {x0.shape}, x0={x0[:7]}...", 2)

        # Open log file if specified
        if self.log_file and not self._log_handle:
            self._log_handle = open(self.log_file, 'w')

        # Call underlying trajOpt
        result = self._planner.trajOpt(vecsPy, N, time_start, time_end, x0, bdotOn)

        # Analyze result
        (success, cost, opt1, lqr_opt, traj_final) = result
        (Xset, Uset, Tset, Kset, Sset, lqr_times) = lqr_opt

        self._log(f"\n  Result: success={success}, final_cost={cost:.6e}", 1)
        self._log(f"  Xset shape: {Xset.shape}, Uset shape: {Uset.shape}", 2)

        # Analyze final controls
        self._print_control_analysis(Uset, prefix="  Final ")

        # Close log file
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        return result

    # =========================================================================
    # Step-by-step methods with debug output
    # =========================================================================

    def prepareForAlilqr(self, vecsPy, dt_use, time_start, time_end, x0, bdotOn):
        """Prepare for ALILQR with debug output."""
        self._log("\n" + "-" * 50, 2)
        self._log("DEBUG: prepareForAlilqr called", 2)

        result = self._planner.prepareForAlilqr(vecsPy, dt_use, time_start, time_end, x0, bdotOn)

        (traj, vecs_dt, costSettings) = result
        (Xset, Uset, Tset, _) = traj

        self._log(f"  Initial trajectory: Xset={Xset.shape}, Uset={Uset.shape}", 2)
        self._print_control_analysis(Uset, prefix="  Initial ")

        return result

    def backwardPass(self, dt, trajPy, vecsPy, auglag_valsPy, regs, costSettings, regSettings, useDist):
        """Backward pass with debug output."""
        self._log("\n  DEBUG: backwardPass", 3)

        result = self._planner.backwardPass(dt, trajPy, vecsPy, auglag_valsPy, regs, costSettings, regSettings, useDist)

        (bp_results, new_regs) = result
        (Kset, dset, Sset) = bp_results

        self._log(f"    dset shape: {dset.shape}", 3)
        self._log(f"    Regs: {regs} -> {new_regs}", 3)

        # Analyze feedforward term (d) for oscillations
        if dset.ndim == 2:
            d_analysis = self._analyze_controls(dset)
            for ch in range(len(d_analysis['sign_changes'])):
                sc = d_analysis['sign_changes'][ch]
                ro = d_analysis['rapid_oscillations'][ch]
                if ro > 5:
                    self._log(f"    WARNING: d[{ch}] has {ro} rapid oscillations!", 2)

        return result

    def forwardPass(self, dt, trajPy, vecsPy, auglag_valsPy, BPresultsPy, regs, costSettings, regSettings, lineSearchSettings, useDist):
        """Forward pass with debug output."""
        self._log("\n  DEBUG: forwardPass", 3)

        result = self._planner.forwardPass(dt, trajPy, vecsPy, auglag_valsPy, BPresultsPy, regs, costSettings, regSettings, lineSearchSettings, useDist)

        (traj_new, newLA, new_regs) = result
        (Xset, Uset, _, _) = traj_new

        self._log(f"    New cost: {newLA:.6e}", 3)
        self._print_control_analysis(Uset, prefix="    ")

        return result

    def maxViol(self, trajPy, vecsPy, auglag_valsPy):
        """Max violation with debug output."""
        result = self._planner.maxViol(trajPy, vecsPy, auglag_valsPy)

        (clist, cmax) = result
        self._log(f"    Max violation: {cmax:.6e}", 2)

        # Analyze constraint activity
        if self.debug_level >= 3 and clist.ndim == 2:
            num_constraints, N = clist.shape
            active_counts = np.sum(clist > 0, axis=1)
            for i in range(num_constraints):
                self._log(f"    Constraint {i}: active at {active_counts[i]}/{N} timesteps", 3)

        return result

    def incrementAugLag(self, auglag_valsPy, clistPy, auglagSettings):
        """Increment augmented Lagrangian with debug output."""
        (lam_old, mu_old, muk_old) = auglag_valsPy

        result = self._planner.incrementAugLag(auglag_valsPy, clistPy, auglagSettings)

        (lam_new, mu_new, muk_new) = result
        self._log(f"  AugLag update: mu {mu_old:.2e} -> {mu_new:.2e}", 2)

        return result

    def ilqrStep(self, dt, trajPy, vecsPy, auglag_valsPy, regs, costSettings, regSettings, lineSearchSettings, breakSettings, useDist):
        """Single iLQR step with debug output."""
        self._log("\n  DEBUG: ilqrStep", 3)

        result = self._planner.ilqrStep(dt, trajPy, vecsPy, auglag_valsPy, regs, costSettings, regSettings, lineSearchSettings, breakSettings, useDist)

        (newLA, cmax, clist, grad, new_regs, traj_new) = result
        self._log(f"    newLA={newLA:.6e}, cmax={cmax:.6e}, grad={grad:.6e}", 3)

        return result

    def alilqr(self, dt, trajPy, vecsPy, costSettings, alilqrSettings, isFirstSearch):
        """Full ALILQR with debug output."""
        self._log("\n" + "-" * 50, 1)
        self._log(f"DEBUG: alilqr (isFirstSearch={isFirstSearch})", 1)

        result = self._planner.alilqr(dt, trajPy, vecsPy, costSettings, alilqrSettings, isFirstSearch)

        (opt_result, final_cost, cmax) = result
        (Xset, Uset, Tset, Kset, Sset, times) = opt_result

        self._log(f"  Final cost: {final_cost:.6e}, max violation: {cmax:.6e}", 1)
        self._print_control_analysis(Uset, prefix="  ")

        return result

    def cost2Func(self, trajPy, vecsPy, auglag_valsPy, costSettings):
        """Cost function with debug output."""
        cost = self._planner.cost2Func(trajPy, vecsPy, auglag_valsPy, costSettings)
        self._log(f"    Cost: {cost:.6e}", 3)
        return cost

    def ilqrBreak(self, grad, LA, dLA, dlaZcount, cmax, iter, breakSettings, outer_iter, ilqr_iter, forOuter):
        """Break condition check with debug output."""
        result = self._planner.ilqrBreak(grad, LA, dLA, dlaZcount, cmax, iter, breakSettings, outer_iter, ilqr_iter, forOuter)
        if result:
            self._log(f"    iLQR break: outer={outer_iter}, inner={ilqr_iter}, grad={grad:.2e}, cmax={cmax:.2e}", 2)
        return result

    def outerBreak(self, auglag_valsPy, cmax, breakSettings, auglagSettings, outer_iter):
        """Outer break condition check with debug output."""
        result = self._planner.outerBreak(auglag_valsPy, cmax, breakSettings, auglagSettings, outer_iter)
        if result:
            self._log(f"    Outer break at iter {outer_iter}, cmax={cmax:.2e}", 1)
        return result

    def generateInitialTrajectory(self, dt, x0, Uset, vecsPy):
        """Generate initial trajectory with debug output."""
        self._log("\n  DEBUG: generateInitialTrajectory", 2)

        result = self._planner.generateInitialTrajectory(dt, x0, Uset, vecsPy)

        (Xset, Uset_out, _, _) = result
        self._log(f"    Generated: Xset={Xset.shape}, Uset={Uset_out.shape}", 2)
        self._print_control_analysis(Uset_out, prefix="    ")

        return result

    def addRandNoise(self, dt, trajPy, dlaZcount, stepsSinceRand, breakSettings, regSettings, costSettings, auglag_vals, vecs):
        """Add random noise with debug output."""
        self._log("    DEBUG: addRandNoise", 3)
        return self._planner.addRandNoise(dt, trajPy, dlaZcount, stepsSinceRand, breakSettings, regSettings, costSettings, auglag_vals, vecs)

    def dynamics(self, x, u, dynamics_info):
        """Dynamics evaluation."""
        return self._planner.dynamics(x, u, dynamics_info)

    def rk4z(self, dt, x, u, dynamics_info_k, dynamics_info_kp1):
        """RK4 integration."""
        return self._planner.rk4z(dt, x, u, dynamics_info_k, dynamics_info_kp1)

    def cleanUpAfterAlilqr(self, vecsPy, dt_prev, time_start, time_end, alilqrOut):
        """Clean up after ALILQR."""
        return self._planner.cleanUpAfterAlilqr(vecsPy, dt_prev, time_start, time_end, alilqrOut)

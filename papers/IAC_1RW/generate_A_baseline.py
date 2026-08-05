"""Campaign A -- baseline grid. Priority 1, the backbone of the paper.

12 cells: {3MTQ+0RW, 3MTQ+1RW, 3MTQ+3RW} x {full attitude, reduced attitude}
          x {PD with LP allocation, planner}

100 paired trials each, one orbit, **both reporting horizons extracted from the same
trajectory** so the 1000 s and one-orbit figures are exactly paired and cost one run.

Nothing else in the paper can be interpreted until these anchor points exist: Campaign E's
sampling plan is built from boundaries computed off A, and the mission-demonstration table is
read straight out of it.

Allocation is pinned to **LP** for every PD cell. Two allocators in the library (``qpc``,
``qpg``) skip the in-pointing torque-free desaturation that LP/QP/QPW perform, so letting one
be picked up would silently confound the allocator with a missing momentum loop across
configurations.

``A_SCALE=fast`` runs a 4-trial, 600 s smoke version.
``A_CELLS=pd`` / ``planner`` restricts which half runs -- the planner cells are orders of
magnitude slower and are usually launched separately.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller import MTQ_w_RW_LP
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from papers.IAC_1RW._iac_sim import (
    HORIZONS_S,
    T_ORBIT,
    cell_metrics,
    make_config,
    simulate,
)

OUT = os.path.join(os.path.dirname(__file__), "output_data")

N_RW = (0, 1, 3)
TASKS = ("reduced", "full")
CONTROLLERS = ("pd", "planner")

# Canonical PD gains, shared across every PD cell so the comparison is law-invariant.
KP, KD, KC = 5e-5, 1e-3, 1e-3

SCALES = {
    "fast":  {"num_runs": 4,   "tf": 1100.0},   # > 1000 s so the metrics path is exercised
    "paper": {"num_runs": 100, "tf": T_ORBIT},
}


def scale() -> Dict[str, Any]:
    return SCALES[os.environ.get("A_SCALE", "paper")]


def make_pd(sat, config):
    return MTQ_w_RW_LP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                       h_target=np.zeros(3))


def make_planner(sat, config):
    """Plan-and-track with the proven Paper-2 tuning.

    Recipe (learned the hard way in the planner paper, do not simplify):
      * plan past the executed window and execute only the first part -- executing to a plan's
        endpoint produces window-joint spikes as the TVLQR gains shrink;
      * cap the AL/iLQR iteration counts, or rare drifted states grind for minutes;
      * ``trajOpt`` **raises** on non-convergence, so the caller needs a reactive fallback.
    """
    from ADCS.controller import Plan_and_Track_LQR
    from ADCS.controller.plan_and_track import CostWeights, PlannerSettings

    ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)
    ps.verbosity = False
    ps.cost_main.use_full_cost_hessian = True
    ps.pass1.regularization.use_dynamics_hess = 1
    ps.init_traj.bdot_gain = 500
    ps.pass1.aug_lag.penalty_init = 1e-3
    ps.pass1.aug_lag.penalty_scale = 10
    ps.pass1.convergence.max_outer_iter = 10
    ps.pass1.convergence.max_inner_iter = 25
    ps.pass2.aug_lag.penalty_init = 1e5
    ps.pass2.aug_lag.penalty_scale = 10
    ps.pass2.convergence.max_outer_iter = 6
    ps.pass2.convergence.max_inner_iter = 15
    ps.cost_main = CostWeights(
        angle=1e1, angle_N=1e1, ang_vel=1e5, ang_vel_N=1e5,
        ang_vel_err_dir=1e2, ang_vel_err_dir_N=0.0, ang_vel_mag=0.0,
        ang_vel_mag_N=0.0, control_mult=1.0, ang_cost_func_type=2)
    ps.cost_second = ps.cost_main
    ps.cost_tvlqr = CostWeights(
        angle=1e5, angle_N=1e6, ang_vel=1e6, ang_vel_N=1e8,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0, control_mult=1.0,
        ang_cost_func_type=2)
    return Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)


MAKERS = {"pd": make_pd, "planner": make_planner}


def _worker(config):
    return simulate(config, MAKERS[config["controller"]])


def cells_to_run() -> List[Dict[str, Any]]:
    want = os.environ.get("A_CELLS", "all")
    ctrls = CONTROLLERS if want == "all" else (want,)
    return [{"n_rw": n, "task": t, "controller": c}
            for c in ctrls for n in N_RW for t in TASKS]


def main() -> int:
    s = scale()
    n, tf = s["num_runs"], s["tf"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    cells = cells_to_run()

    print("=" * 82)
    print(f"Campaign A -- baseline grid: {len(cells)} cells x {n} trials, tf = {tf:.0f} s")
    print(f"horizons reported: {', '.join(f'{h:.0f} s' for h in HORIZONS_S)}")
    print("=" * 82)

    results: Dict[str, Any] = {}
    for cell in cells:
        key = f"{cell['n_rw']}rw_{cell['task']}_{cell['controller']}"
        cfgs = [dict(make_config(rid, n_rw=cell["n_rw"], task=cell["task"],
                                 tf=tf, dt=1.0, seed=rid),
                     controller=cell["controller"])
                for rid in range(n)]
        print(f"\n[{key}] running {n} trials...")
        t0 = time.time()
        runner = MonteCarloRunner(sim_func=_worker,
                                  config_generator=lambda i, _c=cfgs: _c[i],
                                  num_runs=len(cfgs))
        runs = [r for r in runner.run() if r is not None]
        el = time.time() - t0

        per_h = {}
        for h in HORIZONS_S:
            if h > tf:
                continue
            m = cell_metrics(runs, h)
            per_h[f"{h:.0f}"] = m
            print(f"  t={h:6.0f}s  conv5 {m['conv_pct_5deg']:5.1f}%  "
                  f"conv1 {m['conv_pct_1deg']:5.1f}%  median {m['median_final_deg']:7.2f} deg  "
                  f"held-p95 {m['median_held_p95_deg']:7.2f} deg")
        results[key] = {"cell": cell, "n_completed": len(runs),
                        "wall_s": el, "horizons": per_h}
        print(f"  ({len(runs)}/{n} completed in {el/60:.1f} min)")

        # Checkpoint after every cell: this campaign runs for hours and a crash in cell 9
        # should not cost cells 1-8.
        with open(f"{OUT}/A_baseline_{ts}.json", "w") as f:
            json.dump({"task": "A_baseline", "timestamp": ts, "n_trials": n,
                       "tf_s": tf, "gains": [KP, KD, KC],
                       "horizons_s": list(HORIZONS_S), "cells": results}, f, indent=2)

    # ---- summary table -------------------------------------------------------------
    print("\n" + "=" * 82)
    for h in HORIZONS_S:
        hk = f"{h:.0f}"
        if not any(hk in v["horizons"] for v in results.values()):
            continue
        print(f"\nHorizon {h:.0f} s")
        print(f"{'cell':<22}{'conv@5':>9}{'conv@1':>9}{'median':>10}{'held p95':>10}"
              f"{'h/hmax':>9}{'MTQ duty':>10}{'trk avail':>11}")
        print("-" * 82)
        for key, v in results.items():
            m = v["horizons"].get(hk)
            if not m or not m.get("n"):
                continue
            hp = m.get("median_peak_h_frac")
            du = m.get("mean_mtq_duty")
            tr = m.get("mean_tracker_available")
            print(f"{key:<22}{m['conv_pct_5deg']:>8.1f}%{m['conv_pct_1deg']:>8.1f}%"
                  f"{m['median_final_deg']:>10.2f}{m['median_held_p95_deg']:>10.2f}"
                  f"{(f'{hp:.3f}' if hp is not None else '-'):>9}"
                  f"{(f'{du:.3f}' if du is not None else '-'):>10}"
                  f"{(f'{tr:.3f}' if tr is not None else '-'):>11}")
    print("=" * 82)
    print(f"\nwrote {OUT}/A_baseline_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

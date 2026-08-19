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

# Gains: inertia-scaled per genACS's own kp ~ ||J|| rule (trace ratio 5.8 from the 3U bus),
# critically damped on the largest transverse inertia. NOT swept-for: the alpha-based
# selection rule was tried and withdrawn (mean alpha does not discriminate on this bus), and
# the sweep showed divergence rising with kp while the median falls -- 2.9e-4 is the largest
# gain with 0% divergence on the reduced task at n=8, and it is the principled value.
J_TRANS = 0.13
KP = 2.9e-4
KD = 2.0 * np.sqrt(KP * J_TRANS / 2.0)
KC = 1e-3

SCALES = {
    "fast":  {"num_runs": 4,   "tf": 1100.0},   # > 1000 s so the metrics path is exercised
    "paper": {"num_runs": 100, "tf": T_ORBIT},
}

#: Trials are weighted by what carries the argument, not spread uniformly.
#:
#: A convergence fraction at n=30 has sigma ~ 5.5% near p=0.9, so +/-11% at 2 sigma -- that
#: cannot resolve a planner-vs-PD gap the companion paper puts at 4 points, and the headline
#: claim would rest on a number too noisy to defend. The 3+1 cells are where the argument
#: lives and get the full count; 3+0 and 3+3 are context (nobody will contest that three
#: wheels converge, or that magnetorquers alone struggle on full attitude) and get fewer.
#:
#: The planner cells are ~10-100x the cost of PD, so this allocation is what makes the
#: comparison affordable at all.
TRIALS_FULL = 100      # 3MTQ+1RW, both tasks
TRIALS_CONTEXT = 30    # 3MTQ+0RW and 3MTQ+3RW


def scale() -> Dict[str, Any]:
    return SCALES[os.environ.get("A_SCALE", "paper")]


def trials_for(cell: Dict[str, Any], default: int) -> int:
    """Full count on the cells that carry the argument, reduced on the context cells."""
    if default < TRIALS_CONTEXT:          # fast/smoke scale: honour it verbatim
        return default
    return TRIALS_FULL if cell["n_rw"] == 1 else TRIALS_CONTEXT


def make_pd(sat, config):
    from papers.IAC_1RW._feedforward import FeedforwardLP
    h0 = np.asarray(config["h0"], float)
    h_t = (h0[0] * np.array([0.0, 0.0, 1.0])) if h0.size else np.zeros(3)
    return FeedforwardLP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                         h_target=h_t, mode="dipole")


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
    # TRACKER WEIGHTS FROZEN FOR CAMPAIGN A. The plan-vs-executed covariate already
    # suggests these track ~1 deg loose, and the temptation once A's planner cells confirm
    # it will be to retune immediately. Do not: a mid-campaign retune makes early and late
    # planner cells disagree with each other and with the PD half they are compared against.
    # Policy: run the half frozen; if the loose-tracking read confirms, retune and rerun the
    # TWO 3+1 PLANNER CELLS ONLY (~5 h at the measured 16 min/trial). The paper then has
    # both numbers -- planner-as-configured and planner-with-tracking-fixed -- and "the gap
    # was tracking, not planning" is demonstrable rather than asserted.
    ps.cost_tvlqr = CostWeights(
        angle=1e5, angle_N=1e6, ang_vel=1e6, ang_vel_N=1e8,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0, control_mult=1.0,
        ang_cost_func_type=2)
    assert (ps.cost_tvlqr.angle, ps.cost_tvlqr.ang_vel) == (1e5, 1e6), (
        "TVLQR weights drifted mid-campaign -- frozen for A; retune only as the "
        "post-A rerun of the two 3+1 planner cells")
    # The planner's authority margin is RELATIVE (umax = control_limit_scale * act.u_max,
    # read from est_sat at construction), so it cannot go stale when the bus changes -- but
    # the reserve constant itself is bus policy, so pin it the same way the bus is pinned.
    assert np.isclose(ps.control_limit_scale, 0.75), (
        f"planner authority reserve drifted: {ps.control_limit_scale} != 0.75")
    assert np.isclose(ps.umax[0], 0.75 * sat.actuators[0].u_max), (
        "planner umax no longer derives from the est_sat actuators")
    return Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)


MAKERS = {"pd": make_pd, "planner": make_planner}


def _worker(config):
    return simulate(config, MAKERS[config["controller"]],
                    disturbances=("gg", "drag", "srp", "dipole", "general"),
                    bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})


def cells_to_run() -> List[Dict[str, Any]]:
    want = os.environ.get("A_CELLS", "all")
    ctrls = CONTROLLERS if want == "all" else (want,)
    return [{"n_rw": n, "task": t, "controller": c}
            for c in ctrls for n in N_RW for t in TASKS]


def main() -> int:
    from papers.IAC_1RW._iac_sim import assert_settled_bus
    assert_settled_bus()
    s = scale()
    n, tf = s["num_runs"], s["tf"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    cells = cells_to_run()

    print("=" * 82)
    alloc = ", ".join(f"{c['n_rw']}rw/{c['task'][:3]}={trials_for(c, n)}" for c in cells)
    print(f"Campaign A -- baseline grid: {len(cells)} cells, tf = {tf:.0f} s")
    print(f"trials per cell: {alloc}")
    print(f"horizons reported: {', '.join(f'{h:.0f} s' for h in HORIZONS_S)}")
    print("=" * 82)

    results: Dict[str, Any] = {}
    for cell in cells:
        key = f"{cell['n_rw']}rw_{cell['task']}_{cell['controller']}"
        n_cell = trials_for(cell, n)
        cfgs = [dict(make_config(rid, n_rw=cell["n_rw"], task=cell["task"],
                                 tf=tf, dt=1.0, seed=rid),
                     controller=cell["controller"])
                for rid in range(n_cell)]
        print(f"\n[{key}] running {n_cell} trials...")
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
            ee = m.get("median_est_att_err_deg")
            tr = m.get("mean_tracker_available")
            print(f"  t={h:6.0f}s  conv5 {m['conv_pct_5deg']:5.1f}%  "
                  f"conv1 {m['conv_pct_1deg']:5.1f}%  median {m['median_final_deg']:7.2f} deg  "
                  f"held-p95 {m['median_held_p95_deg']:7.2f} deg  "
                  f"KNOWLEDGE {('%.3f' % ee) if ee is not None else '-':>7} deg  "
                  f"trk {('%.2f' % tr) if tr is not None else '-'}")
            if ee is not None and ee > 1.0:
                print(f"           ^ knowledge floor {ee:.2f} deg exceeds the 1 deg "
                      f"threshold -- conv1 for this cell measures the FILTER, not actuation")
        results[key] = {"cell": cell, "n_completed": len(runs),
                        "wall_s": el, "horizons": per_h}
        print(f"  ({len(runs)}/{n_cell} completed in {el/60:.1f} min)")

        # Checkpoint after every cell: this campaign runs for hours and a crash in cell 9
        # should not cost cells 1-8.
        with open(f"{OUT}/A_baseline_{ts}.json", "w") as f:
            json.dump({"task": "A_baseline", "timestamp": ts, "n_trials": n,
                       "trials_full": TRIALS_FULL, "trials_context": TRIALS_CONTEXT,
                       "tf_s": tf, "gains": [KP, KD, KC],
                       "horizons_s": list(HORIZONS_S), "cells": results}, f, indent=2)

    # ---- paired planner-vs-PD deltas -----------------------------------------------
    #
    # Report these, not the difference of two independent convergence proportions. The seeds
    # are paired by construction, so the per-trial error difference and the win fraction have
    # far lower variance than a difference of proportions -- at n=30 that is the difference
    # between a defensible claim and a shrug. Median and mean final error are continuous and
    # hold up much better than a threshold-crossing fraction.
    paired: Dict[str, Any] = {}
    for n_rw in N_RW:
        for task in TASKS:
            kp = f"{n_rw}rw_{task}_pd"
            kq = f"{n_rw}rw_{task}_planner"
            if kp not in results or kq not in results:
                continue
            for hk in (f"{h:.0f}" for h in HORIZONS_S):
                mp = results[kp]["horizons"].get(hk)
                mq = results[kq]["horizons"].get(hk)
                if not mp or not mq or not mp.get("n") or not mq.get("n"):
                    continue
                a = np.asarray(mp["finals_deg"], float)
                b = np.asarray(mq["finals_deg"], float)
                m = min(a.size, b.size)          # paired by seed = by index
                if m < 2:
                    continue
                d = a[:m] - b[:m]                # positive => planner better
                sem = float(np.std(d, ddof=1) / np.sqrt(m))
                paired[f"{n_rw}rw_{task}@{hk}"] = {
                    "n_paired": int(m),
                    "median_delta_deg": float(np.median(d)),
                    "mean_delta_deg": float(np.mean(d)),
                    "sem_delta_deg": sem,
                    "planner_win_frac": float(np.mean(d > 0.0)),
                    "pd_median_deg": float(np.median(a[:m])),
                    "planner_median_deg": float(np.median(b[:m])),
                }

    if paired:
        print("\n" + "=" * 82)
        print("Paired planner-vs-PD (positive delta = planner better; same seeds)")
        print(f"{'cell@horizon':<24}{'n':>5}{'med d':>9}{'mean d':>9}{'SEM':>8}"
              f"{'win frac':>10}{'PD med':>9}{'plan med':>10}")
        print("-" * 82)
        for k, v in paired.items():
            print(f"{k:<24}{v['n_paired']:>5}{v['median_delta_deg']:>9.2f}"
                  f"{v['mean_delta_deg']:>9.2f}{v['sem_delta_deg']:>8.2f}"
                  f"{v['planner_win_frac']:>10.2f}{v['pd_median_deg']:>9.2f}"
                  f"{v['planner_median_deg']:>10.2f}")

    # ---- summary table -------------------------------------------------------------
    print("\n" + "=" * 82)
    for h in HORIZONS_S:
        hk = f"{h:.0f}"
        if not any(hk in v["horizons"] for v in results.values()):
            continue
        print(f"\nHorizon {h:.0f} s")
        print(f"{'cell':<22}{'conv@5':>9}{'conv@1':>9}{'median':>10}{'held p95':>10}"
              f"{'know':>8}{'h peak':>8}{'h end':>8}{'duty':>7}{'trk':>6}")
        print("-" * 82)
        for key, v in results.items():
            m = v["horizons"].get(hk)
            if not m or not m.get("n"):
                continue
            hp = m.get("median_peak_h_frac")
            hf = m.get("median_final_h_frac")
            du = m.get("mean_mtq_duty")
            tr = m.get("mean_tracker_available")
            ee = m.get("median_est_att_err_deg")
            print(f"{key:<22}{m['conv_pct_5deg']:>8.1f}%{m['conv_pct_1deg']:>8.1f}%"
                  f"{m['median_final_deg']:>10.2f}{m['median_held_p95_deg']:>10.2f}"
                  f"{(f'{ee:.2f}' if ee is not None else '-'):>8}"
                  f"{(f'{hp:.3f}' if hp is not None else '-'):>8}"
                  f"{(f'{hf:.3f}' if hf is not None else '-'):>8}"
                  f"{(f'{du:.2f}' if du is not None else '-'):>7}"
                  f"{(f'{tr:.2f}' if tr is not None else '-'):>6}")
    print("=" * 82)
    with open(f"{OUT}/A_baseline_{ts}.json", "w") as f:
        json.dump({"task": "A_baseline", "timestamp": ts, "n_trials": n,
                   "trials_full": TRIALS_FULL, "trials_context": TRIALS_CONTEXT,
                   "tf_s": tf, "gains": [KP, KD, KC],
                   "horizons_s": list(HORIZONS_S), "cells": results,
                   "paired_planner_vs_pd": paired}, f, indent=2)
    print(f"\nwrote {OUT}/A_baseline_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

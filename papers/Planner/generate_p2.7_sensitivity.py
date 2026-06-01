"""P2.7 -- Planner hyperparameter sensitivity sweep (Paper 2, nice-to-have).

One-at-a-time sweep around the canonical planner tuning, on 3+1 / reduced
(vector-pointing) MC, measuring how convergence, mean final pointing error and
planner solve time respond to:
  * pass1 augmented-Lagrangian penalty_init  in {1e-4, 1e-3*, 1e-2}
  * cost_main.angle (attitude weight)          in {1, 10*, 100}
  * cost_main.control_mult (effort weight)      in {0.5, 1.0*, 2.0}
(* = baseline; the baseline is run once and shared across dimensions.)

Small by design (nice-to-have): N trials per setting via P27_TRIALS (default 6;
PAPER2_SCALE=fast forces 2). Emits:
  output_data/P2.7_sensitivity_<ts>.json
  output_data/fig_sensitivity.{png,pdf}
  P2.7_RESULTS.md (separate).
"""

import os
import sys
import json
import time
import datetime as _dt

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
CONFIG = "3+1"
BASE_SEED = 42
CONV_DEG = 5.0

BASE = {"penalty_init": 1e-3, "angle": 10.0, "control_mult": 1.0}
SWEEP = {
    "penalty_init": [1e-4, 1e-3, 1e-2],
    "angle": [1.0, 10.0, 100.0],
    "control_mult": [0.5, 1.0, 2.0],
}


def make_settings(sat, penalty_init, angle, control_mult):
    ps = P.make_planner_settings(sat)
    ps.pass1.aug_lag.penalty_init = penalty_init
    cw = ADCS.controller.plan_and_track.CostWeights
    ps.cost_main = cw(angle=angle, angle_N=1e1, ang_vel=1e5, ang_vel_N=1e5,
                      ang_vel_err_dir=1e2, ang_vel_err_dir_N=0.0,
                      ang_vel_mag=0.0, ang_vel_mag_N=0.0,
                      control_mult=control_mult, ang_cost_func_type=2)
    ps.cost_second = ps.cost_main
    return ps


def mc_config():
    return ADCS.MCConfig(
        w=lambda rng: ADCS.helpers.normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
        h=lambda rng: rng.uniform(-1e-4, 1e-4, size=1),
        goal=lambda rng: ADCS.goals.ECI_Goal(
            eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))),
        orbit=P.make_random_os)


def _bore(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        if isinstance(b, dict):
            b = b.get("default", next(iter(b.values())))
    b = np.asarray(b, float).reshape(3)
    return b / np.linalg.norm(b)


def evaluate(params, n_trials):
    """Run a small MC at the given params; return conv%, mean final err, and a
    single-plan solve time."""
    sat = P.make_sat(CONFIG, estimated=False)
    ps = make_settings(sat, **params)
    ctrl = ADCS.controller.Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    os0 = P.default_os0()

    # time one representative plan
    rng = np.random.default_rng(BASE_SEED)
    gl = ADCS.GoalList(goal_timeline={0.0: ADCS.goals.ECI_Goal(
        ADCS.helpers.normalize(rng.standard_normal(3)))},
        time_units="seconds", start_juliantime=0.22)
    t0 = time.perf_counter()
    ctrl.calculate_trajectory(0.22, 1000.0, P.x0(1), os0, gl)
    solve_s = time.perf_counter() - t0

    # small MC for convergence (fresh controller each campaign)
    ctrl2 = ADCS.controller.Plan_and_Track_LQR(
        est_sat=P.make_sat(CONFIG, estimated=False), planner_settings=ps)
    res = ADCS.simulate_mc(
        x=P.x0(1), satellite=P.make_sat(CONFIG, estimated=False),
        controller=ctrl2, goal=ADCS.goals.ECI_Goal(np.array([1.0, 0, 0])),
        os0=os0, dt=1.0, tf=1000.0, mc_config=mc_config(),
        num_runs=n_trials, base_seed=BASE_SEED)
    bore = _bore(res.satellite)
    finals = []
    for run in res.runs:
        st = np.asarray(run.state_hist, float)
        tg = np.asarray(run.target_hist, float)
        finals.append(_angle_deg(_boresight_eci(st[-1, 3:7], bore), tg[-1][1:4]))
    finals = np.asarray(finals)
    return {"conv_pct": float(100 * np.mean(finals < CONV_DEG)),
            "mean_final_deg": float(np.mean(finals)),
            "median_final_deg": float(np.median(finals)),
            "solve_s": float(solve_s), "n": int(finals.size)}


def main():
    ts = os.environ.get("P27_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    n = 2 if os.environ.get("PAPER2_SCALE") == "fast" else \
        int(os.environ.get("P27_TRIALS", "6"))
    print(f"[P2.7] config={CONFIG} n_trials/setting={n} ts={ts}")

    base_metrics = evaluate(BASE, n)
    print(f"  baseline: conv {base_metrics['conv_pct']:.0f}% mean "
          f"{base_metrics['mean_final_deg']:.2f} solve {base_metrics['solve_s']:.1f}s")
    results = {"baseline": {"params": BASE, **base_metrics}, "dimensions": {}}

    for dim, values in SWEEP.items():
        pts = []
        for v in values:
            if v == BASE[dim]:
                m = base_metrics
            else:
                params = dict(BASE); params[dim] = v
                m = evaluate(params, n)
                print(f"  {dim}={v}: conv {m['conv_pct']:.0f}% mean "
                      f"{m['mean_final_deg']:.2f} solve {m['solve_s']:.1f}s")
            pts.append({"value": v, **m})
        results["dimensions"][dim] = pts

    # figure: 3 columns (dims) x 2 rows (mean err, solve time)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for j, (dim, pts) in enumerate(results["dimensions"].items()):
        xs = [p["value"] for p in pts]
        axs[0, j].plot(xs, [p["mean_final_deg"] for p in pts], "o-")
        axs[0, j].set_title(dim); axs[0, j].set_ylabel("mean final err [deg]")
        axs[1, j].plot(xs, [p["solve_s"] for p in pts], "s-", color="C1")
        axs[1, j].set_ylabel("plan solve [s]"); axs[1, j].set_xlabel(dim)
        if dim in ("penalty_init",):
            axs[0, j].set_xscale("log"); axs[1, j].set_xscale("log")
        for ax in (axs[0, j], axs[1, j]):
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"P2.7: planner hyperparameter sensitivity ({CONFIG}, n={n}/setting)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sensitivity.png", dpi=150)
    fig.savefig(f"{OUT}/fig_sensitivity.pdf")
    plt.close(fig)

    payload = {"task": "P2.7_sensitivity", "timestamp": ts, "config": CONFIG,
               "n_trials_per_setting": n, "conv_threshold_deg": CONV_DEG,
               **results}
    with open(f"{OUT}/P2.7_sensitivity_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P2.7] wrote {OUT}/P2.7_sensitivity_{ts}.json + fig_sensitivity")
    return payload


if __name__ == "__main__":
    main()

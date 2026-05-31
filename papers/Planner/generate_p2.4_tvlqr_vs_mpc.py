"""P2.4 -- TVLQR vs single-step MPC trajectory tracking (Paper 2).

Quantifies the trajectory-tracking difference between the TVLQR feedback law
(``Plan_and_Track_LQR``) and the single-step MPC tracker
(``Plan_and_Track_SingleStepMPC``).  Both controllers subclass the same
planner and share the *identical* ALTRO/SALTRO planning path, so the only
difference is the per-step tracking law:

  * TVLQR : ``u = u_ref - K dx`` (linear feedback, post-hoc saturation clip)
  * MPC   : bounded 1-step QP minimising the K-gain-weighted next-step error
            with a forward RK4 prediction at the *actual* state/B-field
            (S ~ K^T K surrogate for the value-function Hessian).

Paired design: both campaigns use the same ``base_seed`` and ``MCConfig`` so
trial ``i`` sees the same IC, goal and orbit under both trackers -> paired
deltas.

Run scale via ``PAPER2_SCALE`` (``fast`` = smoke, ``paper`` = published 100).

Outputs (papers/Planner/output_data/):
  * P2.4_tvlqr_vs_mpc_<ts>.json   -- per-trial + aggregate payload
  * P2.4_pointing_error.{png,pdf} -- mean pointing error vs time, both trackers
  * P2.4_paired_delta.{png,pdf}   -- paired final-error + effort scatter
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
from ADCS.helpers import metrics as M
import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
os.makedirs(OUT, exist_ok=True)

CONFIG_KEY = "3+1"            # consistent with P2.1 / P2.2
BASE_SEED = 42
CONV_THRESH_DEG = 5.0


# --------------------------------------------------------------------------- #
# Shared scenario (vector-pointing / "reduced" task, the canonical Paper 2 MC)
# --------------------------------------------------------------------------- #
def make_mc_config():
    """Identical scenario distribution for both trackers (paired)."""
    return ADCS.MCConfig(
        w=lambda rng: ADCS.helpers.normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
        h=lambda rng: rng.uniform(-1e-4, 1e-4, size=1),
        goal=lambda rng: ADCS.goals.ECI_Goal(
            eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))),
        orbit=P.make_random_os,
    )


def run_campaign(controller_factory):
    """Run one paired MC campaign; return SimulationResults."""
    real_sat = P.make_sat(CONFIG_KEY, estimated=False)
    n_rw = len([a for a in real_sat.actuators
                if a.__class__.__name__ == "RW"])
    ctrl = controller_factory(real_sat)
    s = P.scale()
    return ADCS.simulate_mc(
        x=P.x0(n_rw), satellite=real_sat, controller=ctrl,
        goal=ADCS.goals.ECI_Goal(eci_vector=np.array([1.0, 0.0, 0.0])),
        os0=P.default_os0(), dt=s["dt"], tf=s["tf"],
        mc_config=make_mc_config(), num_runs=s["num_runs"],
        base_seed=BASE_SEED)


# --------------------------------------------------------------------------- #
# Per-trial metric extraction
# --------------------------------------------------------------------------- #
def per_trial(results):
    dicts = M.from_simulation_results(results)
    out = []
    for i, d in enumerate(dicts):
        t, err = M.run_pointing_error(d)
        u = np.asarray(d.get("u"), dtype=float) if d.get("u") is not None \
            else np.zeros((len(t), 0))
        # control effort: time-integrated L2 norm of the command
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        eff = float(np.sum(np.linalg.norm(u, axis=1)) * dt) if u.size else 0.0
        out.append({
            "trial": i,
            "final_error_deg": float(err[-1]),
            "rms_error_deg": float(np.sqrt(np.mean(err ** 2))),
            "mean_error_deg": float(np.mean(err)),
            "settling_time_s": M.settling_time(t, err, CONV_THRESH_DEG),
            "control_effort": eff,
            "converged": bool(err[-1] < CONV_THRESH_DEG),
            # downsampled error curve for the paper figure
            "t_ds": t[::5].tolist(),
            "err_ds": err[::5].tolist(),
        })
    return out


# --------------------------------------------------------------------------- #
# Per-step solve-time micro-benchmark (single representative trajectory)
# --------------------------------------------------------------------------- #
def solve_time_benchmark(tf_bench=120.0):
    """Time ``find_u`` per step inside a real single-trajectory sim for each
    tracker (identical scenario), by monkeypatching the controller's find_u
    with a timing wrapper.  Returns per-step solve-time stats (ms)."""
    from ADCS.mc.simulate_mc import _simulate_with_precomputed_orbit
    from ADCS.orbits.universal_constants import TimeConstants
    from ADCS.orbits.orbit import Orbit

    rng = np.random.default_rng(BASE_SEED)
    os0 = P.default_os0()
    goal = ADCS.goals.ECI_Goal(
        eci_vector=ADCS.helpers.normalize(rng.standard_normal(3)))
    x = P.x0(1)
    x[3:7] = ADCS.helpers.normalize(rng.standard_normal(4))
    dt = 1.0
    N = int(tf_bench / dt)
    sec2cent = TimeConstants.sec2cent
    orb = Orbit(os0=os0, end_time=os0.J2000 + (tf_bench + 5) * sec2cent,
                dt=dt, use_J2=True, fast=False, verbose=False)
    os_seq = [orb.get_os(J2000=os0.J2000 + k * dt * sec2cent)
              for k in range(N + 1)]

    stats = {}
    for name, fac in (("tvlqr", lambda s: ADCS.controller.Plan_and_Track_LQR(
                            est_sat=s, planner_settings=P.make_planner_settings(s))),
                      ("mpc", lambda s: ADCS.controller.Plan_and_Track_SingleStepMPC(
                            est_sat=s, planner_settings=P.make_planner_settings(s)))):
        sat = P.make_sat(CONFIG_KEY, estimated=False)
        ctrl = fac(sat)
        dts = []
        _orig = ctrl.find_u

        def timed(*a, **k):
            t0 = time.perf_counter()
            out = _orig(*a, **k)
            dts.append((time.perf_counter() - t0) * 1e3)
            return out

        ctrl.find_u = timed
        _simulate_with_precomputed_orbit(
            x=x, satellite=sat, est_satellite=None, controller=ctrl,
            estimator=None, orbit_estimator=None, goal=goal,
            os_seq=os_seq, dt=dt, tf=tf_bench)
        dts = np.asarray(dts) if dts else np.array([np.nan])
        stats[name] = {
            "mean_ms": float(np.nanmean(dts)),
            "median_ms": float(np.nanmedian(dts)),
            "p95_ms": float(np.nanpercentile(dts, 95)),
            "max_ms": float(np.nanmax(dts)),
            "n_calls": int(np.sum(np.isfinite(dts))),
        }
    return stats


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_pointing_error(tv, mp, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for trials, label, color in ((tv, "TVLQR", "C0"), (mp, "MPC", "C1")):
        # interpolate each trial's downsampled curve onto a common grid
        grids = [np.asarray(d["t_ds"]) for d in trials]
        tg = grids[0]
        E = np.vstack([np.interp(tg, np.asarray(d["t_ds"]),
                                 np.asarray(d["err_ds"])) for d in trials])
        ax.plot(tg, np.mean(E, axis=0), color=color, lw=2, label=f"{label} mean")
        ax.fill_between(tg, np.percentile(E, 5, axis=0),
                        np.percentile(E, 95, axis=0), color=color, alpha=0.15)
    ax.axhline(CONV_THRESH_DEG, ls="--", c="k", lw=0.8,
               label=f"{CONV_THRESH_DEG:g} deg threshold")
    ax.set_xlabel("time [s]"); ax.set_ylabel("pointing error [deg]")
    ax.set_yscale("log"); ax.set_title("P2.4: TVLQR vs single-step MPC tracking")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


def fig_paired(tv, mp, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    fe_t = np.array([d["final_error_deg"] for d in tv])
    fe_m = np.array([d["final_error_deg"] for d in mp])
    ef_t = np.array([d["control_effort"] for d in tv])
    ef_m = np.array([d["control_effort"] for d in mp])
    for ax, a, b, name in ((axs[0], fe_t, fe_m, "final pointing error [deg]"),
                           (axs[1], ef_t, ef_m, "control effort")):
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        ax.scatter(a, b, s=18, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"TVLQR {name}"); ax.set_ylabel(f"MPC {name}")
        ax.grid(True, alpha=0.3)
    axs[0].set_title("Paired final error"); axs[1].set_title("Paired effort")
    fig.suptitle("P2.4: paired TVLQR vs MPC (points below diagonal = MPC better)")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ts = os.environ.get("P24_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P2.4] scale={os.environ.get('PAPER2_SCALE','fast')} ts={ts}")

    print("[P2.4] TVLQR campaign ...")
    tv_res = run_campaign(lambda s: ADCS.controller.Plan_and_Track_LQR(
        est_sat=s, planner_settings=P.make_planner_settings(s)))
    print("[P2.4] MPC campaign ...")
    mp_res = run_campaign(lambda s: ADCS.controller.Plan_and_Track_SingleStepMPC(
        est_sat=s, planner_settings=P.make_planner_settings(s)))

    tv = per_trial(tv_res); mp = per_trial(mp_res)

    print("[P2.4] solve-time benchmark ...")
    try:
        solve = solve_time_benchmark()
    except Exception as e:  # pragma: no cover
        solve = {"error": repr(e)}

    # ---- aggregates + paired deltas (TVLQR - MPC) ---- #
    def arr(trials, k): return np.array([d[k] for d in trials], dtype=float)
    paired = {}
    for k in ("final_error_deg", "rms_error_deg", "control_effort",
              "settling_time_s"):
        a, b = arr(tv, k), arr(mp, k)
        d = a - b
        paired[k] = {
            "tvlqr_mean": float(np.nanmean(a)), "mpc_mean": float(np.nanmean(b)),
            "tvlqr_p95": float(np.nanpercentile(a, 95)),
            "mpc_p95": float(np.nanpercentile(b, 95)),
            "delta_mean_tvlqr_minus_mpc": float(np.nanmean(d)),
            "delta_median": float(np.nanmedian(d)),
        }
    agg = {
        "tvlqr_conv_pct": float(100 * np.mean(arr(tv, "converged"))),
        "mpc_conv_pct": float(100 * np.mean(arr(mp, "converged"))),
        "paired": paired,
        "solve_time": solve,
    }

    payload = {
        "task": "P2.4_tvlqr_vs_mpc", "timestamp": ts,
        "config": CONFIG_KEY, "base_seed": BASE_SEED,
        "scale": os.environ.get("PAPER2_SCALE", "fast"),
        "n_trials": len(tv), "conv_threshold_deg": CONV_THRESH_DEG,
        "trackers": {"tvlqr": "Plan_and_Track_LQR",
                     "mpc": "Plan_and_Track_SingleStepMPC (S~K^T K)"},
        "aggregate": agg,
        "per_trial": {"tvlqr": tv, "mpc": mp},
    }
    jpath = os.path.join(OUT, f"P2.4_tvlqr_vs_mpc_{ts}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P2.4] wrote {jpath}")

    fig_pointing_error(tv, mp, os.path.join(OUT, "P2.4_pointing_error"))
    fig_paired(tv, mp, os.path.join(OUT, "P2.4_paired_delta"))
    print("[P2.4] figures written")

    # headline to stdout
    print("\n=== P2.4 HEADLINE ===")
    print(f" TVLQR conv {agg['tvlqr_conv_pct']:.0f}%  | MPC conv {agg['mpc_conv_pct']:.0f}%")
    for k, v in paired.items():
        print(f" {k:18s}: TVLQR {v['tvlqr_mean']:.4g}  MPC {v['mpc_mean']:.4g} "
              f" (Δ mean {v['delta_mean_tvlqr_minus_mpc']:+.4g})")
    if "tvlqr" in solve:
        print(f" solve/step: TVLQR {solve['tvlqr']['median_ms']:.3f} ms  "
              f"MPC {solve['mpc']['median_ms']:.3f} ms")
    return payload


if __name__ == "__main__":
    main()

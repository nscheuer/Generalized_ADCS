"""P2.3 -- Sequential multi-goal Monte Carlo (Paper 2).

The planner chains several terminal pointing goals without intermediate
replanning or mode switching: point at A, hold, slew to B, hold, slew to C.
Complements P2.2 (spinning terminal state): together they support the
"arbitrary terminal targets" claim.

Scenario (the existing generate_altro_*_mixed timeline):
  t in [  0, 300): ECI_Goal A   (acquire + hold)
  t in [300, 400): No_Goal      (coast)
  t in [400, 700): ECI_Goal B   (slew + hold)
  t in [700, 800): No_Goal      (coast)
  t in [800,1000]: ECI_Goal C   (slew + hold)
A,B,C are random unit ECI directions per trial (paired across configs).

Configs: 3+1 and 3+3. 100 trials, paired seeds.  PAPER2_SCALE=paper.

Emits per config:
  output_data/P2.3_multigoal_<config>_<ts>.json   (per-trial + aggregate)
  output_data/P2.3_multigoal_<config>_<ts>.sim    (raw, re-extraction)
  output_data/fig_multigoal_<config>.{png,pdf}     (representative trial)
  P2.3_RESULTS.md written separately.
"""

import os
import sys
import json
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
os.makedirs(OUT, exist_ok=True)

BASE_SEED = 42
CONV_DEG = 5.0
# Symmetric window structure: A active | no-goal | B active | no-goal | C active
# with equal active widths (G) and equal no-goal coast widths (C). Widened from
# the original 300/100 to 500/200 (v6) so the first slew (goal A) can complete.
ACTIVE_W = float(os.environ.get("P23_ACTIVE_W", 500.0))
COAST_W = float(os.environ.get("P23_COAST_W", 200.0))
_G, _C = ACTIVE_W, COAST_W
# goal A:[0,G) B:[G+C,2G+C) C:[2G+2C,3G+2C); coasts in between
WINDOWS = [("A", 0.0, _G), ("B", _G + _C, 2 * _G + _C),
           ("C", 2 * _G + 2 * _C, 3 * _G + 2 * _C)]
CHANGE_TIMES = [0.0, _G, _G + _C, 2 * _G + _C, 2 * _G + 2 * _C]
TF = 3 * _G + 2 * _C


def goallist_for(rng):
    A = ADCS.helpers.normalize(rng.standard_normal(3))
    B = ADCS.helpers.normalize(rng.standard_normal(3))
    C = ADCS.helpers.normalize(rng.standard_normal(3))
    timeline = {0.0: ADCS.goals.ECI_Goal(A), _G: ADCS.goals.No_Goal(),
                _G + _C: ADCS.goals.ECI_Goal(B), 2 * _G + _C: ADCS.goals.No_Goal(),
                2 * _G + 2 * _C: ADCS.goals.ECI_Goal(C)}
    return ADCS.GoalList(goal_timeline=timeline, time_units="seconds",
                         start_juliantime=0.22)


def make_mc_config():
    return ADCS.MCConfig(
        w=lambda rng: ADCS.helpers.normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
        h=lambda rng: rng.uniform(-1e-4, 1e-4, size=1) if False else None,
        goal=lambda rng: goallist_for(rng),
        orbit=P.make_random_os,
    )


def run_config(config_key):
    real_sat = P.make_sat(config_key, estimated=False)
    n_rw = len([a for a in real_sat.actuators if a.__class__.__name__ == "RW"])
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=real_sat, planner_settings=P.make_planner_settings(real_sat))
    s = P.scale()
    # h sampler depends on n_rw; rebuild mc_config with correct h size
    mc = ADCS.MCConfig(
        w=lambda rng: ADCS.helpers.normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
        h=lambda rng, _n=n_rw: rng.uniform(-1e-4, 1e-4, size=_n),
        goal=lambda rng: goallist_for(rng),
        orbit=P.make_random_os,
    )
    n = int(os.environ.get("P23_TRIALS", s["num_runs"]))
    return ADCS.simulate_mc(
        x=P.x0(n_rw), satellite=real_sat, controller=ctrl,
        goal=goallist_for(np.random.default_rng(0)),
        os0=P.default_os0(), dt=s["dt"], tf=TF,
        mc_config=mc, num_runs=n, base_seed=BASE_SEED)


def _resolve_boresight(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        if isinstance(b, dict):
            b = b.get("default", next(iter(b.values())))
    b = np.asarray(b, float).reshape(3)
    return b / np.linalg.norm(b)


def per_window_errors(run, bore_unit):
    """Pointing error (deg) vs the active goal, per timestep + per window."""
    st = np.asarray(run.state_hist, float)
    tg = np.asarray(run.target_hist, float)
    t = np.asarray(run.time_s, float)
    err = np.full(len(st), np.nan)
    for k in range(len(st)):
        target = tg[k]
        if np.any(np.isnan(target)) and target.size == 4 and np.isnan(target[0]):
            tv = target[1:4]
            if np.linalg.norm(tv) == 0 or np.any(np.isnan(tv)):
                continue
            err[k] = _angle_deg(_boresight_eci(st[k, 3:7], bore_unit), tv)
    win = {}
    for label, t0, t1 in WINDOWS:
        m = (t >= t0) & (t < t1)
        e = err[m]
        e = e[np.isfinite(e)]
        if e.size:
            win[label] = {
                "final_in_window_deg": float(e[-1]),
                "min_in_window_deg": float(np.min(e)),
                "converged": bool(e[-1] < CONV_DEG),
                "acquired": bool(np.min(e) < CONV_DEG),
            }
        else:
            win[label] = {"final_in_window_deg": float("nan"),
                          "min_in_window_deg": float("nan"),
                          "converged": False, "acquired": False}
    return t, err, win


def analyse(results, config_key):
    bore = _resolve_boresight(results.satellite)
    n_rw = len([a for a in results.satellite.actuators
                if a.__class__.__name__ == "RW"])
    trials = []
    for i, run in enumerate(results.runs):
        t, err, win = per_window_errors(run, bore)
        st = np.asarray(run.state_hist, float)
        h = st[:, 7:7 + n_rw] if n_rw else np.zeros((len(st), 0))
        u = (np.asarray(run.control_hist, float)
             if getattr(run, "control_hist", None) is not None else None)
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        mtq_eff = (float(np.sum(np.abs(u[:, :3])) * dt)
                   if u is not None and u.shape[1] >= 3 else float("nan"))
        trials.append({
            "trial": i, "windows": win,
            "rw_h_max": float(np.max(np.abs(h))) if h.size else 0.0,
            "rw_h_final": h[-1].tolist() if h.size else [],
            "mtq_effort": mtq_eff,
        })
    # aggregate per-goal convergence
    agg = {}
    for label, _, _ in WINDOWS:
        conv = [tr["windows"][label]["converged"] for tr in trials]
        acq = [tr["windows"][label]["acquired"] for tr in trials]
        finals = np.array([tr["windows"][label]["final_in_window_deg"]
                           for tr in trials], float)
        agg[label] = {
            "conv_pct": float(100 * np.mean(conv)),
            "acquired_pct": float(100 * np.mean(acq)),
            "mean_final_deg": float(np.nanmean(finals)),
            "median_final_deg": float(np.nanmedian(finals)),
        }
    all_conv = [all(tr["windows"][l]["converged"] for l, _, _ in WINDOWS)
                for tr in trials]
    agg["all_goals"] = {"conv_pct": float(100 * np.mean(all_conv))}
    return trials, agg, bore


def representative_figure(results, config_key, bore, path):
    """Single-trial: pointing error vs time + RW momentum, goal-change marks."""
    # pick the trial that converges on all goals with smallest mean final error
    best_i, best_score = 0, np.inf
    n_rw = len([a for a in results.satellite.actuators
                if a.__class__.__name__ == "RW"])
    for i, run in enumerate(results.runs):
        _, _, win = per_window_errors(run, bore)
        if all(win[l]["converged"] for l, _, _ in WINDOWS):
            score = np.nanmean([win[l]["final_in_window_deg"]
                                for l, _, _ in WINDOWS])
            if score < best_score:
                best_score, best_i = score, i
    run = results.runs[best_i]
    t, err, _ = per_window_errors(run, bore)
    st = np.asarray(run.state_hist, float)
    h = st[:, 7:7 + n_rw] if n_rw else np.zeros((len(st), 0))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, err, lw=1.5, color="C0")
    ax.axhline(CONV_DEG, ls=":", c="k", lw=0.8)
    for tc in CHANGE_TIMES[1:]:
        ax.axvline(tc, ls="--", c="gray", lw=0.8)
    for label, t0, t1 in WINDOWS:
        ax.axvspan(t0, t1, color="C1", alpha=0.06)
        ax.text((t0 + t1) / 2, ax.get_ylim()[1] * 0.55, f"goal {label}",
                ha="center", fontsize=9, color="C3")
    ax.set_yscale("log"); ax.set_ylabel("pointing error [deg]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"P2.3 multi-goal, {config_key} (trial {best_i}, "
                 f"active={ACTIVE_W:.0f}s coast={COAST_W:.0f}s)")
    ax.grid(True, which="both", alpha=0.3)
    # wheel-momentum inset
    if h.size:
        axin = ax.inset_axes([0.62, 0.58, 0.36, 0.38])
        for j in range(h.shape[1]):
            axin.plot(t, h[:, j], lw=1.0, label=f"RW{j}")
        for tc in CHANGE_TIMES[1:]:
            axin.axvline(tc, ls="--", c="gray", lw=0.5)
        axin.set_title("RW momentum [Nms]", fontsize=7)
        axin.tick_params(labelsize=6)
        if h.shape[1] <= 3:
            axin.legend(fontsize=5, ncol=h.shape[1], loc="upper right")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)
    return best_i


def main():
    ts = os.environ.get("P23_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    configs = os.environ.get("P23_CONFIGS", "3+1,3+3").split(",")
    print(f"[P2.3] configs={configs} active={ACTIVE_W:.0f}s coast={COAST_W:.0f}s "
          f"tf={TF:.0f}s scale={os.environ.get('PAPER2_SCALE','fast')} ts={ts}")
    summary = {}
    table_rows = []   # (config, goal) cells
    for cfg in configs:
        print(f"[P2.3] config {cfg} ...")
        res = run_config(cfg)
        tag = cfg.replace("+", "p")
        try:
            res.save(f"P2.3_multigoal_{tag}_{ts}", out_dir=OUT)
        except Exception as e:
            print(f"[P2.3] warn save: {e!r}")
        trials, agg, bore = analyse(res, cfg)
        best_i = representative_figure(res, cfg, bore,
                                       os.path.join(OUT, f"fig_multigoal_{tag}"))
        payload = {"task": "P2.3_multigoal", "config": cfg, "timestamp": ts,
                   "n_trials": len(trials), "windows": [w[0] for w in WINDOWS],
                   "active_w": ACTIVE_W, "coast_w": COAST_W, "tf": TF,
                   "conv_threshold_deg": CONV_DEG, "aggregate": agg,
                   "representative_trial": best_i, "per_trial": trials}
        with open(os.path.join(OUT, f"P2.3_multigoal_{tag}_{ts}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        summary[cfg] = agg
        for lbl, _, _ in WINDOWS:
            a = agg[lbl]
            table_rows.append({"config": cfg, "goal": lbl, "n": len(trials),
                               "acquired_pct": a["acquired_pct"],
                               "held_pct": a["conv_pct"],
                               "mean_final_deg": a["mean_final_deg"],
                               "median_final_deg": a["median_final_deg"]})
        print(f"  {cfg}: per-goal ACQUIRED " +
              ", ".join(f"{l}={agg[l]['acquired_pct']:.0f}%" for l, _, _ in WINDOWS) +
              " | held " +
              ", ".join(f"{l}={agg[l]['conv_pct']:.0f}%" for l, _, _ in WINDOWS) +
              f" | all-goals-held={agg['all_goals']['conv_pct']:.0f}%")

    # tab_multigoal (6 rows: config x goal)
    cols = ["config", "goal", "n", "acquired_pct", "held_pct",
            "mean_final_deg", "median_final_deg"]
    with open(os.path.join(OUT, "tab_multigoal.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in table_rows:
            f.write(",".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    with open(os.path.join(OUT, "tab_multigoal.tex"), "w") as f:
        f.write("\\begin{tabular}{l l r r r r}\n\\hline\n")
        f.write("Config & Goal & $n$ & Acquired\\% & Held\\% & Mean final (deg) \\\\\n\\hline\n")
        last = None
        for r in table_rows:
            if last is not None and r["config"] != last:
                f.write("\\hline\n")
            last = r["config"]
            f.write(f"{r['config']} & {r['goal']} & {r['n']} & "
                    f"{r['acquired_pct']:.0f} & {r['held_pct']:.0f} & "
                    f"{r['mean_final_deg']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"[P2.3] wrote tab_multigoal.{{tex,csv}}")
    print("[P2.3] done:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

"""P1.2 -- one PD law, multiple platforms (Paper 1, "portability" claim).

The *same* control law (framework LP-PD) and the *same* canonical Paper-1 PD
gains are run on three actuator complements (3MTQ+0RW / 3MTQ+1RW / 3MTQ+3RW)
and two task types (vector-pointing and full 3-axis attitude), 100 paired
trials each. The demonstration is *consistency across configs modulo actuator
authority* -- not absolute pointing performance.

Canonical gains (kp,kd,kc) = (5e-5, 1e-3, 1e-3): identical to the
controllability / allocation (FIG-ALLOC) / SAMELAW figures, so P1.2 inherits a
testbed consistent with the rest of Paper 1.

Framing lock (carried from LP1<->LP2): this is about *relative* behaviour
across configs on the same task; report paired deltas, saturation (1-alpha)
and convergence flagged by controllability appropriateness, not headline
absolute convergence numbers alongside P2.1.

PAPER1_SCALE (fast=smoke, paper=100). Emits:
  output_data/P1.2_same_pd_<ts>.json
  output_data/tab_same_pd.{tex,csv}
  output_data/fig_same_pd.{png,pdf}
  P1.2_RESULTS.md (written separately)
"""

import os
import sys
import json
import datetime as _dt
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers import metrics as M
from ADCS.helpers.math_helpers import normalize
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUT = "papers/Generalized_ACS/output_data"
CONFIGS = ["3MTQ+0RW", "3MTQ+1RW", "3MTQ+3RW"]
TASKS = ["vector", "full"]
CONV_DEG = 5.0
KP, KD, KC = 5e-5, 1e-3, 1e-3   # canonical Paper-1 PD gains

# Controllability appropriateness drawn from Paper 1's controllability tables:
# 3 MTQ give instantaneous rank-2 torque (perp B); RWs add body axes.
#   vector pointing (2-DOF boresight): achievable for all (magnetic suffices).
#   full 3-axis attitude: needs 3 independent axes -> 3+3 appropriate, 3+1
#   marginal (1 wheel + magnetic), 3+0 inappropriate (no full 3-axis authority).
APPROPRIATE = {
    ("3MTQ+0RW", "vector"): True,  ("3MTQ+0RW", "full"): False,
    ("3MTQ+1RW", "vector"): True,  ("3MTQ+1RW", "full"): "marginal",
    ("3MTQ+3RW", "vector"): True,  ("3MTQ+3RW", "full"): True,
}


def make_controller(sat: Satellite, config: Dict[str, Any]):
    return MTQ_w_RW_LP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                       h_target=np.zeros(3))


def _worker(config):
    return simulate(config, make_controller)


def attitude_error_deg(state, q_ref):
    q = np.asarray(state, float)[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    qr = q_ref / np.linalg.norm(q_ref)
    return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(q @ qr), 0.0, 1.0)))


def goal_quat_for(run_id):
    return normalize(np.random.default_rng(20_000 + run_id).standard_normal(4))


def err_series(run, task):
    """Pointing error (deg) time series for one run, per task type."""
    if task == "full":
        qr = np.asarray(run["config"]["goal_quat"], float)
        return np.asarray(run["time"], float), attitude_error_deg(run["state"], qr)
    # vector: Paper-1 boresight metric (sat boresight = [0,0,1] in _paper1_sim)
    return M.run_pointing_error(run)


def saturation(run):
    """Mean realized LP saturation 1 - mean(alpha) over the run (nan if the
    controller did not record alpha)."""
    a = run.get("alpha")
    if a is None:
        return float("nan")
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.0 - np.mean(a)) if a.size else float("nan")


def main():
    s = scale()
    tf, dt, n = s["tf"], s["dt"], s["num_runs"]
    ts = os.environ.get("P12_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P1.2] gains=({KP},{KD},{KC}) tf={tf} dt={dt} n={n} ts={ts}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells: List[Dict[str, Any]] = []
    per_cell_finals: Dict[str, List[float]] = {}

    for task in TASKS:
        for cfg in CONFIGS:
            def gen(rid, _cfg=cfg, _task=task):
                c = make_config(rid, _cfg, tf, dt, seed=rid)  # paired across cfgs
                if _task == "full":
                    c["goal_quat"] = goal_quat_for(rid)
                return c

            runner = MonteCarloRunner(sim_func=_worker, config_generator=gen,
                                      num_runs=n)
            res = [r for r in runner.run() if r is not None]
            if not res:
                raise RuntimeError(f"P1.2 failed cfg={cfg} task={task}")

            finals, sats, settles = [], [], []
            for r in res:
                t, e = err_series(r, task)
                finals.append(float(e[-1]))
                st = M.settling_time(t, e, CONV_DEG)
                if np.isfinite(st):
                    settles.append(st)
                sv = saturation(r)
                if np.isfinite(sv):
                    sats.append(sv)
            finals = np.asarray(finals)
            key = f"{cfg}|{task}"
            per_cell_finals[key] = finals.tolist()
            cells.append({
                "config": cfg, "task": task, "n": int(finals.size),
                "conv_pct": float(100 * np.mean(finals < CONV_DEG)),
                "mean_final_deg": float(np.mean(finals)),
                "p95_final_deg": float(np.percentile(finals, 95)),
                "mean_settle_s": float(np.mean(settles)) if settles else float("nan"),
                "mean_saturation": float(np.mean(sats)) if sats else float("nan"),
                "controllability": APPROPRIATE[(cfg, task)],
            })
            c = cells[-1]
            print(f"  {cfg:9s} {task:6s}: conv {c['conv_pct']:5.1f}%  "
                  f"mean {c['mean_final_deg']:7.2f}  p95 {c['p95_final_deg']:7.2f}  "
                  f"1-a {c['mean_saturation']:.3f}  appropriate={c['controllability']}")

    # paired deltas across configs on the SAME task (vs 3+3 as the most-capable)
    for task in TASKS:
        base = np.asarray(per_cell_finals[f"3MTQ+3RW|{task}"])
        for c in cells:
            if c["task"] == task:
                d = np.asarray(per_cell_finals[f"{c['config']}|{task}"]) - base
                c["paired_delta_vs_3p3_mean_deg"] = float(np.mean(d))

    # figure: mean final error vs config, one line per task
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xpos = {c: i for i, c in enumerate(CONFIGS)}
    for task, mk in (("vector", "o-"), ("full", "s--")):
        xs = [xpos[c["config"]] for c in cells if c["task"] == task]
        ys = [c["mean_final_deg"] for c in cells if c["task"] == task]
        p95 = [c["p95_final_deg"] for c in cells if c["task"] == task]
        ax.plot(xs, ys, mk, label=f"{task} (mean)", lw=1.8)
        ax.fill_between(xs, ys, p95, alpha=0.12)
    ax.set_xticks(range(len(CONFIGS))); ax.set_xticklabels(CONFIGS)
    ax.axhline(CONV_DEG, ls=":", c="k", lw=0.8)
    ax.set_yscale("log"); ax.set_ylabel("final pointing error [deg]")
    ax.set_title(f"P1.2 (FIG-SAMEPD): one PD law across configs, n={n}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_same_pd.png", dpi=150); fig.savefig(f"{OUT}/fig_same_pd.pdf")
    plt.close(fig)

    # tables
    cols = ["config", "task", "n", "conv_pct", "mean_final_deg", "p95_final_deg",
            "mean_settle_s", "mean_saturation", "controllability",
            "paired_delta_vs_3p3_mean_deg"]
    with open(f"{OUT}/tab_same_pd.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for c in cells:
            f.write(",".join(
                f"{c[k]:.3f}" if isinstance(c.get(k), float) else str(c.get(k, ""))
                for k in cols) + "\n")
    with open(f"{OUT}/tab_same_pd.tex", "w") as f:
        f.write("\\begin{tabular}{l l r r r r l}\n\\hline\n")
        f.write("Config & Task & Conv\\% & Mean & P95 & $1-\\bar\\alpha$ & Controllable \\\\\n\\hline\n")
        for c in cells:
            f.write(f"{c['config']} & {c['task']} & {c['conv_pct']:.0f} & "
                    f"{c['mean_final_deg']:.2f} & {c['p95_final_deg']:.2f} & "
                    f"{c['mean_saturation']:.3f} & {c['controllability']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    payload = {
        "task": "P1.2_same_pd", "timestamp": ts, "n_trials": n,
        "gains": [KP, KD, KC], "conv_threshold_deg": CONV_DEG,
        "tasks": TASKS, "configs": CONFIGS, "cells": cells,
        "per_cell_finals": per_cell_finals,
    }
    with open(f"{OUT}/P1.2_same_pd_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P1.2] wrote {OUT}/P1.2_same_pd_{ts}.json + tab/fig")
    return payload


if __name__ == "__main__":
    main()

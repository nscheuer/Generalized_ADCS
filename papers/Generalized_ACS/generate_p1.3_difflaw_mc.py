"""P1.3 -- four control laws through the framework (Paper 1, Section V-B).

100-trial Monte Carlo on 3MTQ+1RW comparing four control laws routed through
the *same* framework actuator interface (LP torque allocation + torque-free
desaturation, MTQ+RW), differing only in the control LAW:

  * LP-PD      -- linear quaternion PD               (MTQ_w_RW_LP)
  * Wie        -- inertia-weighted eigenaxis regulator (MTQ_Wie)
  * Lovera     -- Lovera-Astolfi PD                  (MTQ_Lovera_LP)
  * Wisniewski -- LTV sliding-mode                    (MTQ_Wisniewski_LP)

All four are framework-allocated (MTQ+RW): the Lovera and Wisniewski *control
laws* are kept but their desired torque is routed through the LP allocation so
the reaction wheel is used (the published versions are magnetic-only and never
command the wheel; use ADCS.controller.MTQ_Lovera / MTQ_Wisniewski for those).
This is the apples-to-apples "swap the control law, one allocation interface"
demonstration: the claim is that the pipeline accepts any law cleanly, not that
one law is best.

Run on BOTH task types: vector-pointing (ECI_Goal, boresight) and full 3-axis
attitude (Fixed_Attitude_Goal). Single knob PAPER1_SCALE (fast=smoke,
paper=100 trials); P13_TF overrides the (longer) sim horizon. Emits:
  output_data/P1.3_difflaw_mc_<ts>.json
  output_data/tab_difflaw_mc.{tex,csv}
  output_data/fig_difflaw_mc.{png,pdf}  (vector | full panels)
  P1.3_RESULTS.md  (written separately)
"""

import os
import sys
import json
import datetime as _dt
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import (MTQ_w_RW_LP, MTQ_Wie, MTQ_Lovera_LP,
                             MTQ_Wisniewski_LP)
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers import metrics as M
from ADCS.helpers.math_helpers import normalize
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUT = "papers/Generalized_ACS/output_data"
CONFIG = "3MTQ+1RW"
CONV_DEG = 5.0
LAWS = ["LP-PD", "Wie", "Lovera", "Wisniewski"]
TASKS = ["vector", "full"]

LP_KP, LP_KD, LP_KC = 5e-5, 1e-3, 1e-3            # canonical Paper-1 PD gains
WIE_KP, WIE_KD = 5e-5 / 0.022, 1e-3 / 0.022       # inertia-weighted (see RESULTS)
LOV_KP, LOV_KD, LOV_EPS = 0.001, 0.005, 1.0
WIS_LS, WIS_LQ = np.diag([0.01] * 3), np.diag([0.002] * 3)


def make_controller(sat: Satellite, config: Dict[str, Any]):
    law = config["law"]
    if law == "LP-PD":
        return MTQ_w_RW_LP(est_sat=sat, p_gain=LP_KP, d_gain=LP_KD,
                           c_gain=LP_KC, h_target=np.zeros(3))
    if law == "Wie":
        return MTQ_Wie(est_sat=sat, p_gain=WIE_KP, d_gain=WIE_KD,
                       c_gain=LP_KC, h_target=np.zeros(3))
    if law == "Lovera":
        return MTQ_Lovera_LP(est_sat=sat, p_gain=LOV_KP, d_gain=LOV_KD,
                             eps=LOV_EPS, c_gain=LP_KC)
    if law == "Wisniewski":
        return MTQ_Wisniewski_LP(est_sat=sat, lambda_s=WIS_LS, lambda_q=WIS_LQ,
                                 c_gain=LP_KC)
    raise ValueError(f"unknown law {law!r}")


def _worker(config):
    return simulate(config, make_controller)


def attitude_error_deg(state, q_ref):
    q = np.asarray(state, float)[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    qr = q_ref / np.linalg.norm(q_ref)
    return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(q @ qr), 0.0, 1.0)))


def goal_quat_for(run_id):
    return normalize(np.random.default_rng(10_000 + run_id).standard_normal(4))


def err_series(run, task):
    if task == "full":
        qr = np.asarray(run["config"]["goal_quat"], float)
        return np.asarray(run["time"], float), attitude_error_deg(run["state"], qr)
    return M.run_pointing_error(run)   # vector: Paper-1 boresight [0,0,1]


def settle(t, err, thr=CONV_DEG):
    above = np.asarray(err) > thr
    if not above.any():
        return float(t[0])
    if above[-1]:
        return float("nan")
    return float(t[int(np.flatnonzero(above)[-1]) + 1])


def main():
    s = scale()
    tf = int(os.environ.get("P13_TF", 2000 if s["num_runs"] > 4 else s["tf"]))
    dt, n = s["dt"], s["num_runs"]
    ts = os.environ.get("P13_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P1.3] {CONFIG} tasks={TASKS} tf={tf} dt={dt} n={n} ts={ts}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {"LP-PD": "C0", "Wie": "C3", "Lovera": "C1", "Wisniewski": "C2"}
    rows: List[Dict[str, Any]] = []
    per_cell: Dict[str, List[float]] = {}

    for ax, task in zip(axs, TASKS):
        for law in LAWS:
            def gen(rid, _law=law, _task=task):
                c = make_config(rid, CONFIG, tf, dt, seed=rid)
                c["law"] = _law
                if _task == "full":
                    c["goal_quat"] = goal_quat_for(rid)
                return c

            runner = MonteCarloRunner(sim_func=_worker, config_generator=gen,
                                      num_runs=n)
            res = [r for r in runner.run() if r is not None]
            if not res:
                raise RuntimeError(f"P1.3 failed law={law} task={task}")
            finals, settles, curves = [], [], []
            for r in res:
                t, err = err_series(r, task)
                finals.append(float(err[-1]))
                st = settle(t, err)
                if np.isfinite(st):
                    settles.append(st)
                curves.append((t, err))
            finals = np.asarray(finals)
            conv = finals < CONV_DEG
            per_cell[f"{task}|{law}"] = finals.tolist()

            tg = curves[0][0]
            E = np.vstack([np.interp(tg, t, e) for t, e in curves])
            ax.plot(tg, np.median(E, axis=0), color=colors[law], lw=1.7, label=law)
            ax.fill_between(tg, np.percentile(E, 25, axis=0),
                            np.percentile(E, 75, axis=0),
                            color=colors[law], alpha=0.10)

            rows.append({
                "task": task, "law": law, "n": int(finals.size),
                "conv_pct": float(100 * np.mean(conv)),
                "mean_final_deg": float(np.mean(finals)),
                "median_final_deg": float(np.median(finals)),
                "p95_final_deg": float(np.percentile(finals, 95)),
                "mean_settle_s": float(np.mean(settles)) if settles else float("nan"),
            })
            print(f"  [{task:6s}] {law:11s}: conv {rows[-1]['conv_pct']:5.1f}% "
                  f"mean {rows[-1]['mean_final_deg']:7.2f} median "
                  f"{rows[-1]['median_final_deg']:7.2f} p95 {rows[-1]['p95_final_deg']:7.2f}")
        ax.axhline(CONV_DEG, ls="--", c="k", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("time [s]")
        ax.set_title(f"{task}-pointing"); ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("pointing / attitude error [deg]"); axs[0].legend(fontsize=8)
    fig.suptitle(f"P1.3 (FIG-DIFFLAW-MC): four laws, {CONFIG} (MTQ+RW), n={n}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_difflaw_mc.png", dpi=150)
    fig.savefig(f"{OUT}/fig_difflaw_mc.pdf"); plt.close(fig)

    # paired delta vs LP-PD within each task
    for r in rows:
        base = np.asarray(per_cell[f"{r['task']}|LP-PD"])
        d = np.asarray(per_cell[f"{r['task']}|{r['law']}"]) - base
        r["paired_delta_vs_LPPD_mean_deg"] = float(np.mean(d))

    cols = ["task", "law", "n", "conv_pct", "mean_final_deg", "median_final_deg",
            "p95_final_deg", "mean_settle_s", "paired_delta_vs_LPPD_mean_deg"]
    with open(f"{OUT}/tab_difflaw_mc.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    with open(f"{OUT}/tab_difflaw_mc.tex", "w") as f:
        f.write("\\begin{tabular}{l l r r r r r r}\n\\hline\n")
        f.write("Task & Law & $n$ & Conv\\% & Mean & Med & P95 & $\\Delta$ vs LP-PD \\\\\n\\hline\n")
        last = None
        for r in rows:
            if last is not None and r["task"] != last:
                f.write("\\hline\n")
            last = r["task"]
            f.write(f"{r['task']} & {r['law']} & {r['n']} & {r['conv_pct']:.0f} & "
                    f"{r['mean_final_deg']:.2f} & {r['median_final_deg']:.2f} & "
                    f"{r['p95_final_deg']:.2f} & {r['paired_delta_vs_LPPD_mean_deg']:+.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    payload = {"task": "P1.3_difflaw_mc", "timestamp": ts, "config": CONFIG,
               "allocation": "framework LP (MTQ+RW) for all four laws",
               "tasks": TASKS, "n_trials": n, "tf": tf, "dt": dt,
               "conv_threshold_deg": CONV_DEG,
               "gains": {"LP-PD": [LP_KP, LP_KD, LP_KC],
                         "Wie": [WIE_KP, WIE_KD, LP_KC],
                         "Lovera": [LOV_KP, LOV_KD, LOV_EPS],
                         "Wisniewski": [0.01, 0.002]},
               "rows": rows, "per_cell_finals": per_cell}
    with open(f"{OUT}/P1.3_difflaw_mc_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P1.3] wrote {OUT}/P1.3_difflaw_mc_{ts}.json + tab/fig")
    return payload


if __name__ == "__main__":
    main()

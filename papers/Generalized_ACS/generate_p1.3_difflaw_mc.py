"""P1.3 -- four control laws through the framework (Paper 1, Section V-B).

100-trial Monte Carlo on 3MTQ+1RW comparing four control laws routed through
the *same* framework actuator interface (LP torque allocation + torque-free
desaturation), differing only in the control LAW:

  * LP-PD      -- linear quaternion PD via the framework LP allocation
                  (this is what the single-run FIG-DIFFLAW previously, and
                  mislabelled, called "Wie")
  * Wie        -- Wie's inertia-weighted eigenaxis quaternion-feedback
                  regulator (tau = -J(kp q_e + kd w_err)); genuinely distinct
                  from LP-PD on the asymmetric inertia (J=diag(.022,.022,.004))
  * Lovera     -- magnetic PD with adaptive projection
  * Wisniewski -- LTV sliding-mode magnetic control

TASK: full-attitude pointing (Fixed_Attitude_Goal, random target quaternion).
A full-attitude task is used so Wie's inertia-weighting is genuinely exercised
on all three axes (the framework's shortest-rotation error handling makes a
plain Wie PD coincide with LP-PD on a boresight task). All four laws are run
on the identical task and the identical paired scenarios (seed = run_id), so
this is the apples-to-apples "swap the control law" demonstration: the claim is
that the pipeline accepts any law cleanly, not that one law is best.

Single knob PAPER1_SCALE (fast=smoke, paper=100 trials). Emits:
  output_data/P1.3_difflaw_mc_<ts>.json
  output_data/tab_difflaw_mc.{tex,csv}
  output_data/fig_difflaw_mc.{png,pdf}
  P1.3_RESULTS.md  (written separately)
"""

import os
import sys
import json
import datetime as _dt
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP, MTQ_Wie, MTQ_Lovera, MTQ_Wisniewski
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers.math_helpers import normalize
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUT = "papers/Generalized_ACS/output_data"
CONFIG = "3MTQ+1RW"
CONV_DEG = 5.0
LAWS = ["LP-PD", "Wie", "Lovera", "Wisniewski"]

# Canonical Paper-1 PD gains (controllability/allocation sections, SAMELAW).
LP_KP, LP_KD, LP_KC = 5e-5, 1e-3, 1e-3
# Wie inertia-weighted gains chosen so J*kp matches LP-PD's kp on the high-
# inertia axes (0.022): kp_wie = 5e-5/0.022, kd_wie = 1e-3/0.022. The low-
# inertia z-axis (0.004) is then driven ~5.5x more gently -> eigenaxis shaping.
WIE_KP, WIE_KD = 5e-5 / 0.022, 1e-3 / 0.022


def make_controller(sat: Satellite, config: Dict[str, Any]):
    law = config["law"]
    if law == "LP-PD":
        return MTQ_w_RW_LP(est_sat=sat, p_gain=LP_KP, d_gain=LP_KD,
                           c_gain=LP_KC, h_target=np.zeros(3))
    if law == "Wie":
        return MTQ_Wie(est_sat=sat, p_gain=WIE_KP, d_gain=WIE_KD,
                       c_gain=LP_KC, h_target=np.zeros(3))
    if law == "Lovera":
        return MTQ_Lovera(est_sat=sat, p_gain=0.001, d_gain=0.005, eps=1.0)
    if law == "Wisniewski":
        return MTQ_Wisniewski(est_sat=sat,
                              lambda_s=np.diag([0.01, 0.01, 0.01]),
                              lambda_q=np.diag([0.002, 0.002, 0.002]))
    raise ValueError(f"unknown law {law!r}")


def _worker(config):
    return simulate(config, make_controller)


def attitude_error_deg(state: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Full-attitude error angle (deg) per timestep: 2*acos(|<q(t), q_ref>|)."""
    q = np.asarray(state, float)[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    qr = q_ref / np.linalg.norm(q_ref)
    dot = np.clip(np.abs(q @ qr), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def settle(t, err, thr=CONV_DEG):
    above = err > thr
    if not above.any():
        return float(t[0])
    if above[-1]:
        return float("nan")
    return float(t[int(np.flatnonzero(above)[-1]) + 1])


def goal_quat_for(run_id: int) -> np.ndarray:
    """Deterministic random target attitude per run_id (paired across laws)."""
    return normalize(np.random.default_rng(10_000 + run_id).standard_normal(4))


def main():
    s = scale()
    tf, dt, n = s["tf"], s["dt"], s["num_runs"]
    ts = os.environ.get("P13_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P1.3] config={CONFIG} task=full-attitude tf={tf} dt={dt} n={n} ts={ts}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    per_law: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    colors = {"LP-PD": "C0", "Wie": "C3", "Lovera": "C1", "Wisniewski": "C2"}

    for law in LAWS:
        def gen(rid, _law=law):
            c = make_config(rid, CONFIG, tf, dt, seed=rid)  # seed=run_id -> paired
            c["law"] = _law
            c["goal_quat"] = goal_quat_for(rid)
            return c

        runner = MonteCarloRunner(sim_func=_worker, config_generator=gen,
                                  num_runs=n)
        res = [r for r in runner.run() if r is not None]
        if not res:
            raise RuntimeError(f"P1.3 run failed for law={law}.")

        finals, settles, curves = [], [], []
        for r in res:
            qr = np.asarray(r["config"]["goal_quat"], float)
            t = np.asarray(r["time"], float)
            err = attitude_error_deg(r["state"], qr)
            finals.append(float(err[-1]))
            st = settle(t, err)
            if np.isfinite(st):
                settles.append(st)
            curves.append((t, err))
        finals = np.asarray(finals)
        conv = finals < CONV_DEG

        # mean error curve on a common grid for the figure
        tg = curves[0][0]
        E = np.vstack([np.interp(tg, t, e) for t, e in curves])
        ax.plot(tg, np.median(E, axis=0), color=colors[law], lw=1.8, label=law)
        ax.fill_between(tg, np.percentile(E, 25, axis=0),
                        np.percentile(E, 75, axis=0), color=colors[law], alpha=0.12)

        per_law[law] = {
            "final_error_deg": finals.tolist(),
            "converged": conv.tolist(),
        }
        rows.append({
            "law": law, "task": "full-attitude", "n": int(finals.size),
            "conv_pct": float(100 * np.mean(conv)),
            "mean_final_deg": float(np.mean(finals)),
            "median_final_deg": float(np.median(finals)),
            "p95_final_deg": float(np.percentile(finals, 95)),
            "mean_settle_s": float(np.mean(settles)) if settles else float("nan"),
        })
        print(f"  {law:11s}: conv {rows[-1]['conv_pct']:5.1f}%  "
              f"mean_final {rows[-1]['mean_final_deg']:7.2f} deg  "
              f"p95 {rows[-1]['p95_final_deg']:7.2f}  "
              f"settle {rows[-1]['mean_settle_s']}")

    # paired deltas vs LP-PD baseline (same scenarios)
    base = np.asarray(per_law["LP-PD"]["final_error_deg"])
    for row in rows:
        d = np.asarray(per_law[row["law"]]["final_error_deg"]) - base
        row["paired_delta_vs_LPPD_mean_deg"] = float(np.mean(d))

    ax.axhline(CONV_DEG, ls="--", c="k", lw=0.8, label=f"{CONV_DEG:g} deg")
    ax.set_yscale("log"); ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title(f"P1.3 (FIG-DIFFLAW-MC): four laws, {CONFIG}, full-attitude, n={n}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_difflaw_mc.png", dpi=150)
    fig.savefig(f"{OUT}/fig_difflaw_mc.pdf")
    plt.close(fig)

    # tables
    cols = ["law", "task", "n", "conv_pct", "mean_final_deg", "median_final_deg",
            "p95_final_deg", "mean_settle_s", "paired_delta_vs_LPPD_mean_deg"]
    with open(f"{OUT}/tab_difflaw_mc.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    with open(f"{OUT}/tab_difflaw_mc.tex", "w") as f:
        f.write("\\begin{tabular}{l l r r r r r r}\n\\hline\n")
        f.write("Law & Task & $n$ & Conv\\% & Mean & Med & P95 & $\\Delta$ vs LP-PD \\\\\n\\hline\n")
        for r in rows:
            f.write(f"{r['law']} & {r['task']} & {r['n']} & {r['conv_pct']:.0f} & "
                    f"{r['mean_final_deg']:.2f} & {r['median_final_deg']:.2f} & "
                    f"{r['p95_final_deg']:.2f} & {r['paired_delta_vs_LPPD_mean_deg']:+.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    payload = {
        "task": "P1.3_difflaw_mc", "timestamp": ts, "config": CONFIG,
        "goal_task": "full-attitude (Fixed_Attitude_Goal)",
        "n_trials": n, "conv_threshold_deg": CONV_DEG,
        "gains": {"LP-PD": [LP_KP, LP_KD, LP_KC],
                  "Wie": [WIE_KP, WIE_KD, LP_KC],
                  "Lovera": [0.001, 0.005, 1.0],
                  "Wisniewski": [0.01, 0.002]},
        "rows": rows, "per_law": per_law,
    }
    with open(f"{OUT}/P1.3_difflaw_mc_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P1.3] wrote {OUT}/P1.3_difflaw_mc_{ts}.json + tab/fig")
    return payload


if __name__ == "__main__":
    main()

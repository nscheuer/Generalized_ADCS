"""
Paper 2 "both for both" 2x2x2 comparison — ingest, tabulate, plot, verify.

Matrix: {3+0 MTQ-only, 3+1} x {reduced-attitude goal, full-attitude goal}
        x {planner (ALTRO), PD baseline}.

Eight pre-existing MC .sim files in papers/Planner/output/ — NO new MC runs.
This script ingests them, applies one locked convergence metric to all 8 cells,
emits a publication table + 2x2 figure + a stats JSON, and asserts the
triage-confirmed cell values (84/27/94/90) reproduce.

LOCKED CONVERGENCE METRIC
-------------------------
* Error quantity: final-timestep pointing error -- state_hist[-1] vs
  target_hist[-1]. This is exactly what the canonical in-codebase histogram
  (ADCS.helpers.plot.control.targetplot.TargetHistogram._get_final_error)
  uses. No steady-state averaging window.
* Full-attitude (quaternion) target rows [q0,q1,q2,q3]:
  _attitude_error_deg -- minimal rotation angle (Hamilton, scalar-first).
* Reduced (vector/ECI) target rows [nan,tx,ty,tz]:
  _angle_deg between the body boresight rotated into ECI and the target
  direction. The .sim files do NOT store boresight_hist, so the body
  boresight is taken from satellite.boresight (= [0,1,0] in all four reduced
  files -- verified). This is the only deviation from _get_final_error and is
  a forced fallback, not a metric choice.
* Convergence threshold: final error < 5 deg (strict), per the paper abstract.

Run:  PYTHONPATH=. venv/bin/python papers/Planner/analyze_p2_matrix.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ADCS
from ADCS.helpers.plot.control.targetplot import (
    _angle_deg, _attitude_error_deg, _boresight_eci,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output_data")
os.makedirs(OUT_DIR, exist_ok=True)

CONV_THRESHOLD_DEG = 5.0  # paper abstract

# ---- the 8 cells: (config, goal, role, controller_label, filename) ----------
CELLS = [
    ("3+0", "reduced", "planner", "ALTRO",  "mc100_altro_3+0_reduced_20260204_185357.sim"),
    ("3+0", "reduced", "pd",      "Lovera", "mc100_lovera_3+0_reduced_20260205_003107.sim"),
    ("3+0", "full",    "planner", "ALTRO",  "mc100_altro_3+0_full_20260204_162440.sim"),
    ("3+0", "full",    "pd",      "Lovera", "mc100_lovera_3+0_full_20260205_003347.sim"),
    ("3+1", "reduced", "planner", "ALTRO",  "mc100_altro_3+1_reduced_20260205_001823.sim"),
    ("3+1", "reduced", "pd",      "LP",     "mc100_lp_3+1_reduced_20260205_000536.sim"),
    ("3+1", "full",    "planner", "ALTRO",  "mc100_altro_3+1_full_20260204_235053.sim"),
    ("3+1", "full",    "pd",      "LP",     "mc100_lp_3+1_full_20260205_000958.sim"),
]

# triage-confirmed regression targets: (config, goal, role) -> (conv%, mean deg)
KNOWN = {
    ("3+0", "reduced", "planner"): (84.0, 3.85),
    ("3+0", "reduced", "pd"):      (27.0, 21.54),
    ("3+1", "full",    "planner"): (94.0, 1.19),
    ("3+1", "full",    "pd"):      (90.0, 9.06),
}


def final_error_deg(run) -> float | None:
    """Locked metric — final-timestep pointing error for one MC run."""
    state_hist = getattr(run, "state_hist", None)
    target_hist = getattr(run, "target_hist", None)
    if state_hist is None or target_hist is None or len(state_hist) == 0:
        return None

    last_state = np.asarray(state_hist[-1])
    last_target = np.asarray(target_hist[-1]).flatten()
    if last_state.size < 7 or last_target.size != 4:
        return None

    q_real = last_state[3:7]

    if not np.isnan(last_target[0]):
        # full-attitude quaternion target
        return _attitude_error_deg(q_real, last_target)

    # reduced / vector ECI target — needs the body boresight
    bore_body = None
    bh = getattr(run, "boresight_hist", None)
    if bh is not None and len(bh) > 0:
        bore_body = np.asarray(bh[-1], dtype=float).flatten()
    if bore_body is None or bore_body.size != 3 or np.linalg.norm(bore_body) == 0:
        sat = getattr(run, "satellite", None)
        b = getattr(sat, "boresight", None)
        if isinstance(b, dict):
            b = list(b.values())[0]
        if b is None:
            return None
        bore_body = np.asarray(b, dtype=float).flatten()
    if bore_body.size != 3 or np.linalg.norm(bore_body) == 0:
        return None

    bore_unit = bore_body / np.linalg.norm(bore_body)
    target_vec = last_target[1:4]
    if np.linalg.norm(target_vec) == 0:
        return None
    target_unit = target_vec / np.linalg.norm(target_vec)
    bore_eci = _boresight_eci(q_real, bore_unit)
    return _angle_deg(bore_eci, target_unit)


def cell_stats(errs: np.ndarray) -> dict:
    return {
        "n": int(errs.size),
        "converged_pct": float(100.0 * np.mean(errs < CONV_THRESHOLD_DEG)),
        "mean_deg": float(np.mean(errs)),
        "median_deg": float(np.median(errs)),
        "p95_deg": float(np.percentile(errs, 95)),
        "min_deg": float(np.min(errs)),
        "max_deg": float(np.max(errs)),
    }


def verify_cell(run0, config: str, role: str, controller_label: str) -> list[str]:
    """Cross-check the loaded .sim content against the filename label."""
    notes = []
    sat = getattr(run0, "satellite", None)
    acts = [type(a).__name__ for a in getattr(sat, "actuators", [])]
    n_rw = acts.count("RW")
    n_mtq = acts.count("MTQ")
    want_rw = 0 if config == "3+0" else 1
    if n_rw != want_rw or n_mtq != 3:
        notes.append(
            f"ACTUATOR MISMATCH: file labelled {config} but satellite has "
            f"{n_mtq} MTQ + {n_rw} RW"
        )
    return notes


def main() -> int:
    ephem = ADCS.Ephemeris()
    rows = []
    stats_json = {
        "metric": {
            "error_quantity": "final-timestep pointing error (state_hist[-1] "
                              "vs target_hist[-1]); no steady-state window",
            "full_attitude": "_attitude_error_deg (minimal rotation angle, "
                             "Hamilton scalar-first)",
            "reduced_attitude": "_angle_deg(boresight-in-ECI, target ECI dir); "
                                "body boresight from satellite.boresight=[0,1,0]",
            "convergence_threshold_deg": CONV_THRESHOLD_DEG,
            "source": "ADCS.helpers.plot.control.targetplot.TargetHistogram",
        },
        "source_branch": "paper2-datagen",
        "cells": {},
    }
    discrepancies = []

    for config, goal, role, ctrl_label, fname in CELLS:
        path = os.path.join(os.path.dirname(__file__), "output", fname)
        res = ADCS.SimulationResults.load(path, ephem=ephem)
        run0 = res.runs[0]

        # cross-check filename vs content
        notes = verify_cell(run0, config, role, ctrl_label)
        # target-type cross-check
        t_last = np.asarray(run0.target_hist[-1]).flatten()
        is_quat = not np.isnan(t_last[0])
        expect_quat = (goal == "full")
        if is_quat != expect_quat:
            notes.append(
                f"GOAL MISMATCH: file labelled {goal} but target rows are "
                f"{'quaternion' if is_quat else 'vector'}"
            )
        n_steps = int(np.asarray(run0.state_hist).shape[0])
        t_end = float(np.asarray(run0.time_s)[-1]) if run0.time_s is not None else None

        errs = np.array(
            [e for e in (final_error_deg(r) for r in res.runs) if e is not None],
            dtype=float,
        )
        st = cell_stats(errs)
        key = f"{config}|{goal}|{role}"
        stats_json["cells"][key] = {
            "config": config, "goal": goal, "role": role,
            "controller": ctrl_label, "file": fname,
            "n_steps": n_steps, "horizon_s": t_end,
            **st, "notes": notes,
        }

        # regression check
        kk = (config, goal, role)
        reg = ""
        if kk in KNOWN:
            exp_pct, exp_mean = KNOWN[kk]
            d_pct = st["converged_pct"] - exp_pct
            d_mean = st["mean_deg"] - exp_mean
            ok = abs(d_pct) <= 0.5 and abs(d_mean) <= 0.05
            reg = "OK" if ok else f"DELTA pct{d_pct:+.2f} mean{d_mean:+.3f}"
            if not ok:
                discrepancies.append(
                    f"{key}: expected {exp_pct}%/{exp_mean}deg, got "
                    f"{st['converged_pct']:.1f}%/{st['mean_deg']:.2f}deg"
                )

        rows.append(dict(config=config, goal=goal, role=role,
                         controller=ctrl_label, n=st["n"],
                         conv=st["converged_pct"], mean=st["mean_deg"],
                         p95=st["p95_deg"], reg=reg, notes=notes,
                         n_steps=n_steps, horizon_s=t_end, errs=errs))
        print(f"  {key:28s} {ctrl_label:7s} n={st['n']:3d} "
              f"conv={st['converged_pct']:6.1f}%  mean={st['mean_deg']:7.2f}  "
              f"p95={st['p95_deg']:7.2f}  {reg}")
        for nt in notes:
            print(f"      ! {nt}")

    _emit_csv(rows)
    _emit_tex(rows)
    _emit_figure(rows)

    stats_json["discrepancies"] = discrepancies
    with open(os.path.join(OUT_DIR, "p2p1_matrix_stats.json"), "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"\n  wrote {OUT_DIR}/p2p1_matrix_stats.json")

    # assertions
    print("\n=== regression check (84/27/94/90) ===")
    if discrepancies:
        for d in discrepancies:
            print("  DISCREPANCY:", d)
        print("  -> known values did NOT all reproduce")
        return 1
    print("  all four known values reproduce within tol (pct +/-0.5, mean +/-0.05)")
    return 0


def _emit_csv(rows):
    path = os.path.join(OUT_DIR, "tab_p2_matrix.csv")
    with open(path, "w") as f:
        f.write("config,goal,role,controller,n,converged_pct,mean_deg,p95_deg,"
                "n_steps,horizon_s,regression\n")
        for r in rows:
            f.write(f"{r['config']},{r['goal']},{r['role']},{r['controller']},"
                    f"{r['n']},{r['conv']:.1f},{r['mean']:.3f},{r['p95']:.3f},"
                    f"{r['n_steps']},{r['horizon_s']:.0f},{r['reg'] or 'n/a'}\n")
    print(f"  wrote {path}")


def _emit_tex(rows):
    """Rows = config; columns = goal; each cell shows planner vs PD."""
    by = {(r["config"], r["goal"], r["role"]): r for r in rows}
    path = os.path.join(OUT_DIR, "tab_p2_matrix.tex")
    lines = [
        "% Paper 2 -- 2x2x2 planner-vs-PD comparison.",
        "% Diagonal (3+0/reduced, 3+1/full) = controllability-appropriate task.",
        "% Cell: convergence% (<5deg) and mean final pointing error [deg].",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Goal formulation} \\",
        r"\cmidrule(lr){3-4}",
        r"Config & Controller & Reduced attitude & Full attitude \\",
        r"\midrule",
    ]
    for config in ("3+0", "3+1"):
        for role, in (("planner",), ("pd",)):
            red = by[(config, "reduced", role)]
            ful = by[(config, "full", role)]
            ctrl = red["controller"]
            label = f"{config} & {ctrl}"
            diag_red = r"\textbf" if (config == "3+0") else ""
            diag_ful = r"\textbf" if (config == "3+1") else ""
            c_red = f"{diag_red}{{{red['conv']:.0f}\\% / {red['mean']:.2f}$^\\circ$}}"
            c_ful = f"{diag_ful}{{{ful['conv']:.0f}\\% / {ful['mean']:.2f}$^\\circ$}}"
            lines.append(f"{label} & {c_red} & {c_ful} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _emit_figure(rows):
    by = {(r["config"], r["goal"], r["role"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    bins = np.arange(0, 95, 5.0)
    for ai, config in enumerate(("3+0", "3+1")):
        for aj, goal in enumerate(("reduced", "full")):
            ax = axes[ai, aj]
            pl = by[(config, goal, "planner")]
            pd = by[(config, goal, "pd")]
            ax.hist(np.clip(pl["errs"], 0, 90), bins=bins, alpha=0.6,
                    label=f"Planner (ALTRO)  {pl['conv']:.0f}%<5deg",
                    color="#2c7fb8", edgecolor="black")
            ax.hist(np.clip(pd["errs"], 0, 90), bins=bins, alpha=0.6,
                    label=f"PD ({pd['controller']})  {pd['conv']:.0f}%<5deg",
                    color="#de2d26", edgecolor="black")
            ax.axvline(CONV_THRESHOLD_DEG, ls="--", color="k", lw=1)
            diag = (config == "3+0" and goal == "reduced") or \
                   (config == "3+1" and goal == "full")
            tag = "  [controllability-appropriate]" if diag else "  [limit case]"
            ax.set_title(f"{config}  /  {goal}-attitude goal{tag}",
                         fontweight="bold" if diag else "normal")
            ax.set_xlabel("Final pointing error [deg]")
            ax.set_ylabel("MC run count")
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("Paper 2 — planner vs PD, both configs x both goal formulations "
                 "(100-trial MC, 1000 s)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT_DIR, "fig_p2_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())

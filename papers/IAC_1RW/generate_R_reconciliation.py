"""Campaign R -- reconciliation against genACS (SSC26-P2-54). Priority 1.

**Purpose: one footnote.** genACS reports 42% convergence for PD on 3MTQ+1RW full attitude at a
1000 s horizon. A reader holding both papers will ask why this paper reports something else. The
answer should be "the horizon", and this campaign is what makes that answer defensible.

Three conditions, 100 paired trials each, on the genACS 3MTQ+1RW full-attitude cell:

===  ==========================================================================
R0   Reproduction. tf = 1000 s, initial rate ~ U(0.1, 2.0) deg/s. Expect ~42%.
R1   Horizon change only: tf = each trial's own orbital period. Nothing else moves.
R2   R1 plus the lower initial-rate spread, ~ U(0.1, 1.0) deg/s.
===  ==========================================================================

**R0 is not optional.** The genACS harness (``papers/Generalized_ACS/_paper1_sim.py``) lives on
an unpushed local branch and forked before several changes to the ADCS core. Running it on
today's core without first reproducing the published number would leave any difference
ambiguous between "the horizon" and "the core moved underneath us" -- which is exactly the
confusion the footnote exists to prevent. R0 answers that first.

Everything genACS did is kept: inertia ``diag(0.022, 0.022, 0.004)``, its actuator scale
(m_max 0.4 A m^2, tau_w 7 mN m, h_max 16.2 mN m s), 2 s control rate, its randomized orbit
(altitude ~ U(400, 1000) km, random inclination/RAAN), **its wheel on +x -- perpendicular to
the boresight** -- truth-state control, and no disturbances at all. The IAC reference bus is
*not* used here, and must not be: growing the actuators or adding the IAC disturbance set would
convert a reconciliation into a goalpost shift.

R2's rate remap is quantile-preserving, so the trials stay paired: a draw at the 30th percentile
of U(0.1, 2.0) maps to the 30th percentile of U(0.1, 1.0), same direction, same everything else.

Run: ``python papers/IAC_1RW/generate_R_reconciliation.py``   (``R_SCALE=fast`` for a smoke run)
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
from ADCS.helpers import metrics as M
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from ADCS.orbits.universal_constants import EarthConstants
from papers.Generalized_ACS._paper1_sim import make_config, simulate


def _load_p1_2():
    """Load genACS's P1.2 generator by path.

    Its filename contains a dot (``generate_p1.2_same_pd.py``) so it is not importable as a
    module path. Loading it anyway -- rather than copying the two functions across -- is the
    point: the reconciliation only means something if the goal draw and the error metric are
    *identical* to the ones that produced the published 42%, and a copy can drift.
    """
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "Generalized_ACS",
                        "generate_p1.2_same_pd.py")
    spec = importlib.util.spec_from_file_location("_genacs_p1_2", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P12 = _load_p1_2()
goal_quat_for = _P12.goal_quat_for          # default_rng(20_000 + run_id)
attitude_error_deg = _P12.attitude_error_deg

OUT = os.path.join(os.path.dirname(__file__), "output_data")

# genACS canonical Paper-1 PD gains (P1.2 / FIG-ALLOC / SAMELAW all share these).
KP, KD, KC = 5e-5, 1e-3, 1e-3
CONV_DEG = 5.0
CONFIG = "3MTQ+1RW"
DT = 2

SCALES = {
    "fast": {"num_runs": 6},
    "paper": {"num_runs": 100},
}


def scale() -> Dict[str, int]:
    return SCALES[os.environ.get("R_SCALE", "paper")]


def orbital_period_s(R_km: np.ndarray) -> float:
    """Circular-orbit period from the trial's own radius.

    genACS randomizes altitude over U(400, 1000) km, so "one orbit" is per-trial: periods run
    5554-6307 s. Holding all trials to a common 5554 s would shorten the high-altitude ones to
    less than a revolution, which is not the comparison the footnote claims to make.
    """
    a = float(np.linalg.norm(R_km))
    return 2.0 * np.pi * np.sqrt(a ** 3 / EarthConstants.mu_e)


def remap_rate_to_narrow(w0: np.ndarray) -> np.ndarray:
    """Quantile-preserving remap of |w0| from U(0.1, 2.0) to U(0.1, 1.0) deg/s.

    Keeps direction and the trial's quantile, so R2 stays paired with R0/R1 rather than being
    an independent draw -- the difference between the cells is then the spread alone.
    """
    deg = np.linalg.norm(w0) * 180.0 / np.pi
    q = (deg - 0.1) / (2.0 - 0.1)
    new_deg = 0.1 + q * (1.0 - 0.1)
    return w0 * (new_deg / deg)


def make_cells(n: int) -> Dict[str, List[Dict[str, Any]]]:
    """Build the three conditions with shared seeds."""
    cells: Dict[str, List[Dict[str, Any]]] = {"R0": [], "R1": [], "R2": []}
    for rid in range(n):
        base = make_config(rid, CONFIG, tf=1000, dt=DT, seed=rid)
        # genACS's full-attitude task, drawn exactly as P1.2 drew it (per-run seed
        # 20_000 + run_id, not one shared stream).
        base["goal_quat"] = goal_quat_for(rid)

        r0 = dict(base)

        r1 = dict(base)
        r1["tf"] = float(orbital_period_s(np.asarray(base["orbit_R"], float)))

        r2 = dict(r1)
        r2["w0"] = remap_rate_to_narrow(np.asarray(base["w0"], float))

        cells["R0"].append(r0)
        cells["R1"].append(r1)
        cells["R2"].append(r2)
    return cells


def make_controller(sat, config):
    return MTQ_w_RW_LP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                       h_target=np.zeros(3))


def _worker(config):
    return simulate(config, make_controller)


def summarize(name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    finals, rates_deg, tfs = [], [], []
    for r in results:
        if r is None:
            continue
        err = attitude_error_deg(r["state"], np.asarray(r["config"]["goal_quat"], float))
        finals.append(float(err[-1]))
        rates_deg.append(float(np.linalg.norm(r["config"]["w0"]) * 180.0 / np.pi))
        tfs.append(float(r["config"]["tf"]))
    finals = np.asarray(finals)
    return {
        "cell": name,
        "n": int(finals.size),
        "conv_pct": float(100.0 * np.mean(finals < CONV_DEG)),
        "mean_final_deg": float(np.mean(finals)),
        "median_final_deg": float(np.median(finals)),
        "p95_final_deg": float(np.percentile(finals, 95)),
        "mean_tf_s": float(np.mean(tfs)),
        "mean_rate_deg_s": float(np.mean(rates_deg)),
        "finals": finals.tolist(),
    }


def main() -> int:
    n = scale()["num_runs"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    cells = make_cells(n)

    print("=" * 78)
    print(f"Campaign R -- genACS reconciliation, {CONFIG} full attitude, n={n}, dt={DT} s")
    print(f"gains ({KP}, {KD}, {KC}); truth state; no disturbances; wheel on +x")
    print("=" * 78)

    out: Dict[str, Any] = {}
    for name in ("R0", "R1", "R2"):
        cfgs = cells[name]
        print(f"\n[{name}] mean tf = {np.mean([c['tf'] for c in cfgs]):.0f} s, "
              f"mean |w0| = {np.mean([np.linalg.norm(c['w0']) for c in cfgs])*180/np.pi:.3f} deg/s")
        runner = MonteCarloRunner(sim_func=_worker,
                                  config_generator=lambda i, _c=cfgs: _c[i],
                                  num_runs=len(cfgs))
        results = [r for r in runner.run() if r is not None]
        out[name] = summarize(name, results)
        c = out[name]
        print(f"[{name}] conv {c['conv_pct']:.1f}%  mean {c['mean_final_deg']:.2f} deg  "
              f"median {c['median_final_deg']:.2f} deg  p95 {c['p95_final_deg']:.2f} deg")

    print("\n" + "=" * 78)
    print(f"{'cell':<6}{'tf [s]':>9}{'rate [deg/s]':>14}{'conv %':>9}{'mean':>9}{'p95':>9}")
    print("-" * 78)
    for name in ("R0", "R1", "R2"):
        c = out[name]
        print(f"{name:<6}{c['mean_tf_s']:>9.0f}{c['mean_rate_deg_s']:>14.3f}"
              f"{c['conv_pct']:>9.1f}{c['mean_final_deg']:>9.2f}{c['p95_final_deg']:>9.2f}")
    print("-" * 78)

    r0 = out["R0"]["conv_pct"]
    print(f"\nPublished genACS value (tab_same_pd.csv, P1.2, tf=1000 s): 42.0%")
    print(f"R0 reproduction on today's core:                          {r0:.1f}%")
    if abs(r0 - 42.0) <= 10.0:
        print("  -> reproduces within 10 points; the core is compatible and the R1/R2")
        print("     deltas below are attributable to the horizon and rate spread.")
    else:
        print("  -> DOES NOT reproduce. The ADCS core has moved since the genACS run, so")
        print("     R1/R2 cannot be attributed to the horizon alone. Reconcile before")
        print("     writing the footnote.")
    print(f"\nHorizon effect  (R1 - R0): {out['R1']['conv_pct'] - r0:+.1f} points")
    print(f"Rate effect     (R2 - R1): {out['R2']['conv_pct'] - out['R1']['conv_pct']:+.1f} points")
    print("=" * 78)

    payload = {"task": "R_reconciliation", "timestamp": ts, "n_trials": n,
               "gains": [KP, KD, KC], "conv_threshold_deg": CONV_DEG,
               "config": CONFIG, "dt": DT, "published_genacs_pct": 42.0,
               "cells": {k: out[k] for k in out}}
    path = f"{OUT}/R_reconciliation_{ts}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

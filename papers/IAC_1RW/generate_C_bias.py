"""Campaign C -- wheel-bias ablation. Isolates gyroscopic stiffness from momentum absorption.

A biased wheel helps a single-wheel bus in two quite different ways, and the paper needs them
separated. **Absorption** is capacity: a wheel with headroom can soak up disturbance momentum.
**Gyroscopic stiffness** is different -- stored momentum ``h`` resists transverse disturbance
torques through ``omega x h``, so a spinning wheel stabilises the two axes it cannot actuate.

The ablation holds **wheel torque and momentum limits fixed** across all four bias levels and
changes only the stored momentum. Any improvement is then unambiguously stiffness, because the
absorption capacity is identical by construction.

Predicted: pointing drift rate falls as ``1/h``.

**Bias is not free**, and the campaign has to say so. A biased wheel makes transverse slews
harder -- reorienting the spacecraft means reorienting a stored angular momentum vector, which
the magnetorquers must fight. So the transverse slew time is measured at each bias level and
reported alongside the drift, otherwise the section reads as though stiffness were costless.

Truth state, full disturbance set, inertial stare, one orbit -- this isolates a mechanism, and
clean isolation is the point (per the campaign spec, C runs without the estimator).

Run: ``python papers/IAC_1RW/generate_C_bias.py``   (``C_SCALE=fast`` for a smoke run)
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
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from ADCS.satellite_factory import IAC_6U
from papers.IAC_1RW._iac_sim import (
    T_ORBIT,
    error_series,
    make_config,
    simulate,
)

OUT = os.path.join(os.path.dirname(__file__), "output_data")

#: Stored momentum as a fraction of h_max. Torque and momentum LIMITS are identical across
#: all four -- only the stored value changes, which is what makes this a stiffness ablation.
BIAS_LEVELS = (0.0, 0.25, 0.50, 0.75)

KP, KD, KC = 5e-5, 1e-3, 1e-3

SCALES = {"fast": {"n": 4, "tf": 900.0}, "paper": {"n": 30, "tf": T_ORBIT}}


def scale():
    return SCALES[os.environ.get("C_SCALE", "paper")]


def make_ctrl(sat, config):
    """PD with the wheel's momentum target set to the trial's bias.

    ``h_target`` is the *commanded* momentum, so the controller holds the bias rather than
    bleeding it off -- without this the desaturation term would drive every cell back to zero
    and the ablation would measure nothing.
    """
    h = np.zeros(3)
    h[:] = np.asarray(config["h_target_vec"], float)
    return MTQ_w_RW_LP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC, h_target=h)


def _worker(config):
    return simulate(config, make_ctrl, use_estimator=False)


def drift_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    """Pointing drift over the held interval, once the initial transient is past."""
    err = error_series(run)
    t = run["time"]
    k0 = int(0.5 * err.size)                       # second half = the held interval
    e, tt = err[k0:], t[k0:]
    if e.size < 3:
        return {"drift_deg_per_orbit": float("nan"), "rms_deg": float("nan"),
                "final_deg": float("nan")}
    slope = float(np.polyfit(tt, e, 1)[0])         # deg/s
    return {"drift_deg_per_orbit": slope * T_ORBIT,
            "rms_deg": float(np.sqrt(np.mean(e ** 2))),
            "final_deg": float(e[-1]),
            "median_deg": float(np.median(e))}


def slew_time(run: Dict[str, Any], thresh_deg: float = 5.0) -> float:
    """Time to first reach ``thresh`` -- the cost side of the bias."""
    err = error_series(run)
    below = np.where(err < thresh_deg)[0]
    return float(run["time"][below[0]]) if below.size else float("nan")


def main() -> int:
    s = scale()
    n, tf = s["n"], s["tf"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 88)
    print(f"Campaign C -- wheel-bias ablation, 3MTQ+1RW, inertial stare, "
          f"n={n}, tf={tf:.0f} s")
    print(f"h_max = {IAC_6U.h_max*1e3:.0f} mN m s (FIXED across all levels; "
          f"only stored momentum varies)")
    print("=" * 88)

    results: Dict[str, Any] = {}
    a_hat = np.asarray(IAC_6U.boresight, float)

    for frac in BIAS_LEVELS:
        h0 = frac * IAC_6U.h_max
        cfgs = []
        for rid in range(n):
            c = make_config(rid, n_rw=1, task="reduced", tf=tf, dt=1.0, seed=rid)
            c["h0"] = np.array([h0])                     # stored momentum along the wheel
            c["h_target_vec"] = h0 * a_hat               # commanded, so it is held not dumped
            cfgs.append(c)

        print(f"\n[h/h_max = {frac:.2f}]  h0 = {h0*1e3:.2f} mN m s, {n} trials...")
        runner = MonteCarloRunner(sim_func=_worker,
                                  config_generator=lambda i, _c=cfgs: _c[i],
                                  num_runs=len(cfgs))
        runs = [r for r in runner.run() if r is not None]

        dm = [drift_metrics(r) for r in runs]
        st = np.array([slew_time(r) for r in runs])
        key = f"{frac:.2f}"
        results[key] = {
            "h_frac": frac, "h0_Nms": h0, "n": len(runs),
            "median_drift_deg_per_orbit": float(np.nanmedian(
                [d["drift_deg_per_orbit"] for d in dm])),
            "median_rms_deg": float(np.nanmedian([d["rms_deg"] for d in dm])),
            "median_final_deg": float(np.nanmedian([d["final_deg"] for d in dm])),
            "median_acquire_5deg_s": float(np.nanmedian(st)),
            "acquired_frac": float(np.mean(np.isfinite(st))),
        }
        r = results[key]
        print(f"  drift {r['median_drift_deg_per_orbit']:+8.3f} deg/orbit   "
              f"RMS {r['median_rms_deg']:7.3f} deg   "
              f"acquire {r['median_acquire_5deg_s']:7.1f} s "
              f"({100*r['acquired_frac']:.0f}% acquired)")

    # ---- does drift fall as 1/h? ------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'h/h_max':>9}{'drift[deg/orbit]':>19}{'RMS[deg]':>11}"
          f"{'acquire[s]':>13}{'x vs h=0':>11}")
    print("-" * 88)
    base = results["0.00"]
    for frac in BIAS_LEVELS:
        r = results[f"{frac:.2f}"]
        rel = (abs(base["median_rms_deg"] / r["median_rms_deg"])
               if r["median_rms_deg"] > 0 else float("nan"))
        print(f"{frac:>9.2f}{r['median_drift_deg_per_orbit']:>19.3f}"
              f"{r['median_rms_deg']:>11.3f}{r['median_acquire_5deg_s']:>13.1f}"
              f"{rel:>11.2f}")

    nz = [f for f in BIAS_LEVELS if f > 0]
    rms = np.array([results[f"{f:.2f}"]["median_rms_deg"] for f in nz])
    hs = np.array([f * IAC_6U.h_max for f in nz])
    ok = np.isfinite(rms) & (rms > 0)
    if ok.sum() >= 2:
        slope = float(np.polyfit(np.log(hs[ok]), np.log(rms[ok]), 1)[0])
        print(f"\n  log(RMS) vs log(h) slope = {slope:+.3f}   (drift ~ 1/h predicts -1)")
    print("\n  Bias is not free: compare the acquire column -- stored momentum has to be")
    print("  reoriented during a transverse slew, and the magnetorquers must fight it.")
    print("=" * 88)

    payload = {"task": "C_bias", "timestamp": ts, "n_trials": n, "tf_s": tf,
               "h_max_Nms": IAC_6U.h_max, "bias_levels": list(BIAS_LEVELS),
               "gains": [KP, KD, KC], "cells": results}
    with open(f"{OUT}/C_bias_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}/C_bias_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

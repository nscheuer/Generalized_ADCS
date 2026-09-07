"""Verification: does the settled bus produce sane results on 3+0, 3+1 and 3+3?

Run after the estimator/gain/mounting fixes and before re-running the campaigns, so that a
bad number is caught here rather than eight hours into Campaign A.

Settled configuration: two star trackers on +-x, tactical-grade gyro, 0.6 A m^2
magnetorquers, the ORIGINAL 2 mN m / 15 mN m s wheel (the upgrade was measured and found not
load-bearing), critically-damped gains at kp = 2e-3, dipole feedforward, and a momentum bias
below the tau_mtq/omega ceiling.

Reports the **boresight-projected** knowledge error, not the 3-axis one. The 3-axis figure
includes roll, which the pointing metric cannot see, and comparing the two directly is what
previously made pointing look better than knowledge.

Expected, if the bus is healthy:
  3+3  full authority                  -> sub-0.05 deg, knowledge-limited
  3+1  the paper's configuration       -> within a few x of 3+3
  3+0  magnetorquer-only               -> converges on the reduced task, fails on full
       (a TIMESCALE result, not a rank one -- genACS Table 4 has 3MTQ+0RW controllable
        for vector pointing under both frozen and time-varying fields)

Run: ``python papers/IAC_1RW/verify_bus.py``
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concurrent.futures import ProcessPoolExecutor

from papers.IAC_1RW._feedforward import FeedforwardLP
from papers.IAC_1RW._iac_sim import (
    T_ORBIT, cell_metrics, make_config, simulate,
)
from ADCS.satellite_factory import IAC_6U

OUT = os.path.join(os.path.dirname(__file__), "output_data")

J_TRANS = 0.13
KP = 2e-3
KD = 2.0 * np.sqrt(KP * J_TRANS / 2.0)
KC = 1e-3
WHEEL = (2.0e-3, 15.0e-3)          # original wheel; the upgrade was not load-bearing
H_FRAC = 0.05                      # below the tau_mtq / omega_slew ceiling
DIST = ("gg", "drag", "srp", "dipole", "general")
N = int(os.environ.get("VERIFY_N", "8"))
NW = int(os.environ.get("NW", "8"))


def _worker(cfg: Dict[str, Any]):
    def mk(sat, config):
        h0 = np.asarray(config["h0"], float)
        # 3MTQ+0RW has no wheel, so there is no momentum target to hold.
        h_t = (h0[0] * np.array([0.0, 0.0, 1.0])) if h0.size else np.zeros(3)
        return FeedforwardLP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                             h_target=h_t, mode="dipole")
    return simulate(cfg, mk, disturbances=DIST,
                    bus_kwargs={"tau_w": WHEEL[0], "h_max": WHEEL[1]})


def _job(a):
    n_rw, task, rid = a
    c = make_config(rid, n_rw=n_rw, task=task, tf=T_ORBIT, dt=1.0, seed=rid)
    c["h0"] = np.full(n_rw, H_FRAC * WHEEL[1])
    try:
        return (n_rw, task, _worker(c))
    except Exception as exc:
        return (n_rw, task, None)


def main() -> int:
    from papers.IAC_1RW._iac_sim import assert_settled_bus
    assert_settled_bus()
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    print("=" * 104)
    print(f"Bus verification -- 3+0 / 3+1 / 3+3, n={N}, one orbit")
    print(f"2 trackers, wheel {WHEEL[0]*1e3:.0f}/{WHEEL[1]*1e3:.0f} mN m(.s), "
          f"m_max {IAC_6U.m_max} A m^2, kp={KP:.0e} kd={KD:.2e} (zeta=1), "
          f"dipole feedforward, bias {H_FRAC:.2f} h_max")
    print("=" * 104)

    jobs = [(n, t, r) for n in (3, 1, 0) for t in ("reduced", "full")
            for r in range(N)]
    buckets: Dict[Any, List] = {}
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for n_rw, task, run in ex.map(_job, jobs):
            if run is not None:
                buckets.setdefault((n_rw, task), []).append(run)

    print(f"\n{'cell':<14}{'n':>3}{'conv5':>8}{'conv1':>8}{'conv.1':>8}"
          f"{'med':>9}{'p95':>9}{'KNOWbore':>10}{'KNOWp95':>9}{'trk':>6}{'hend':>7}")
    print("-" * 104)
    out: Dict[str, Any] = {}
    for n_rw in (3, 1, 0):
        for task in ("reduced", "full"):
            runs = buckets.get((n_rw, task), [])
            if not runs:
                print(f"{f'{n_rw}RW {task}':<14}  (no runs completed)")
                continue
            m = cell_metrics(runs, T_ORBIT)
            key = f"{n_rw}rw_{task}"
            out[key] = m
            f = lambda v, p=4: "-" if v is None else f"{v:.{p}f}"
            print(f"{f'{n_rw}RW {task}':<14}{m['n']:>3}"
                  f"{m['conv_pct_5deg']:>7.0f}%{m['conv_pct_1deg']:>7.0f}%"
                  f"{100*np.mean(np.asarray(m['finals_deg'])<0.1):>7.0f}%"
                  f"{m['median_final_deg']:>9.4f}{m['p95_final_deg']:>9.4f}"
                  f"{f(m.get('median_bore_knowledge_deg')):>10}"
                  f"{f(m.get('p95_bore_knowledge_deg')):>9}"
                  f"{f(m.get('mean_tracker_available'),2):>6}"
                  f"{f(m.get('median_final_h_frac'),3):>7}")

    with open(f"{OUT}/verify_bus_{ts}.json", "w") as fh:
        json.dump({"task": "verify_bus", "timestamp": ts, "n": N,
                   "kp": KP, "kd": KD, "wheel": list(WHEEL), "h_frac": H_FRAC,
                   "m_max": IAC_6U.m_max, "cells": out}, fh, indent=2)
    print("=" * 104)
    print(f"\nwrote {OUT}/verify_bus_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

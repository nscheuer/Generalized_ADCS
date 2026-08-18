"""Gain SELECTION for the 3MTQ+1RW bus, using the LP scale factor as the tuning signal.

Not a tuning check -- a selection, and a methodological point the paper can carry.

**Why the campaign had this wrong.** kp was chosen by sweeping on 3RW, where the wheels
absorb whatever the allocator asks for, and the resulting number was then ported to a bus
where torque is roughly 300x scarcer. genACS SS V-A documents the failure directly: the LP
folds feedback and feedforward into a single scale factor, so a large reference command from
aggressive gains drives ``alpha`` down and starves *everything* -- not just the excess. Per
configuration gains through inertia-and-authority scaling is genACS's own thesis, and tuning
on one complement and carrying the number violates it.

Measured at kp = 2e-3 on 3+1: **alpha median ~0.01**. The allocator was delivering about 1%
of the requested torque.

**The selection rule.** ``alpha`` is the natural tuning signal for an LP-allocated
underactuated bus, and it does not appear to be used that way anywhere. Pick the largest kp
whose mean ``alpha`` stays above ~0.5 outside the initial transient -- i.e. the stiffest
controller that does not spend its life saturating the allocator. If that holds up, "on an
underactuated bus, tune until the allocator stops saturating" is a practitioner rule.

**The tension is real and is part of the result.** The steady-state disturbance floor is
``tau_dist / kp``, which pushes kp *up*; alpha pushes it *down*. Where those conflict on a
3+1 bus is itself informative, so both are reported at every gain rather than optimising one.

Run: ``python papers/IAC_1RW/select_gains_3plus1.py``
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from papers.IAC_1RW._feedforward import FeedforwardLP
from papers.IAC_1RW._iac_sim import T_ORBIT, error_series, make_config, simulate

OUT = os.path.join(os.path.dirname(__file__), "output_data")

J_TRANS = 0.13
WHEEL = (2.0e-3, 15.0e-3)
H_FRAC = 0.05
DIST = ("gg", "drag", "srp", "dipole", "general")
TAU_DIST = 2.5e-6                     # peak disturbance torque, for the tau/kp floor
ALPHA_TARGET = 0.5

KPS = (5e-5, 1.5e-4, 2.9e-4, 6e-4, 1.2e-3, 2e-3)
N = int(os.environ.get("GAIN_N", "8"))
NW = int(os.environ.get("NW", "10"))


def _job(a):
    kp, task, rid = a
    kd = 2.0 * np.sqrt(kp * J_TRANS / 2.0)

    def mk(sat, cfg):
        h0 = np.asarray(cfg["h0"], float)
        h_t = (h0[0] * np.array([0.0, 0.0, 1.0])) if h0.size else np.zeros(3)
        return FeedforwardLP(est_sat=sat, p_gain=kp, d_gain=kd, c_gain=1e-3,
                             h_target=h_t, mode="dipole")

    c = make_config(rid, n_rw=1, task=task, tf=T_ORBIT, dt=1.0, seed=rid)
    c["h0"] = np.full(1, H_FRAC * WHEEL[1])
    try:
        r = simulate(c, mk, disturbances=DIST,
                     bus_kwargs={"tau_w": WHEEL[0], "h_max": WHEEL[1]})
    except Exception:
        return None
    e = error_series(r)
    h = len(e) // 2                        # "outside the initial transient"
    al = r["alpha"][h:]
    al = al[np.isfinite(al)]
    return dict(kp=kp, task=task, rid=rid,
                final=float(e[-1]), med=float(np.median(e[h:])),
                p95=float(np.percentile(e[h:], 95)),
                alpha_mean=float(np.mean(al)) if al.size else np.nan,
                alpha_med=float(np.median(al)) if al.size else np.nan,
                alpha_lowfrac=float(np.mean(al < 0.5)) if al.size else np.nan,
                sigma_med=float(np.median(r["sigma"][h:])),
                hmax=float(np.max(r["h_frac"])))


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    print("=" * 104)
    print(f"Gain selection on 3MTQ+1RW -- alpha as the tuning signal. n={N}, one orbit.")
    print(f"Rule: largest kp with mean alpha > {ALPHA_TARGET} outside the transient.")
    print(f"Tension: the tau_dist/kp floor pushes kp UP, alpha pushes it DOWN.")
    print("=" * 104)

    jobs = [(kp, t, r) for kp in KPS for t in ("reduced", "full") for r in range(N)]
    rows = [x for x in ProcessPoolExecutor(max_workers=NW).map(_job, jobs) if x]

    res: Dict[str, Any] = {}
    for task in ("reduced", "full"):
        print(f"\n--- 3+1 {task} ---")
        print(f"{'kp':>9}{'kd':>10}{'a_mean':>8}{'a_med':>8}{'a<0.5':>7}"
              f"{'med[deg]':>10}{'p95[deg]':>10}{'diverged':>10}{'tau/kp':>9}")
        print("-" * 90)
        for kp in KPS:
            v = [x for x in rows if x["kp"] == kp and x["task"] == task]
            if not v:
                continue
            am = np.nanmean([x["alpha_mean"] for x in v])
            div = np.mean([x["final"] > 10 for x in v])
            key = f"{task}|{kp:.1e}"
            res[key] = {"kp": kp, "task": task, "n": len(v),
                        "alpha_mean": float(am),
                        "alpha_median": float(np.nanmedian([x["alpha_med"] for x in v])),
                        "alpha_frac_below_0p5": float(np.nanmean([x["alpha_lowfrac"] for x in v])),
                        "median_deg": float(np.median([x["med"] for x in v])),
                        "p95_deg": float(np.median([x["p95"] for x in v])),
                        "diverged_frac": float(div),
                        "tau_over_kp_deg": float(np.degrees(TAU_DIST / kp)),
                        "per_trial": v}
            r_ = res[key]
            print(f"{kp:>9.1e}{2*np.sqrt(kp*J_TRANS/2):>10.2e}{am:>8.3f}"
                  f"{r_['alpha_median']:>8.3f}{r_['alpha_frac_below_0p5']:>7.2f}"
                  f"{r_['median_deg']:>10.4f}{r_['p95_deg']:>10.3f}"
                  f"{100*div:>9.0f}%{r_['tau_over_kp_deg']:>9.3f}")

    # ---- apply the rule ----------------------------------------------------------------
    print("\n" + "=" * 104)
    ok = [res[f"reduced|{kp:.1e}"] for kp in KPS
          if f"reduced|{kp:.1e}" in res
          and res[f"reduced|{kp:.1e}"]["alpha_mean"] > ALPHA_TARGET]
    if ok:
        pick = max(ok, key=lambda r: r["kp"])
        print(f"SELECTED kp = {pick['kp']:.2e}  (kd = {2*np.sqrt(pick['kp']*J_TRANS/2):.3e})")
        print(f"  mean alpha {pick['alpha_mean']:.3f} > {ALPHA_TARGET}; "
              f"median {pick['median_deg']:.4f} deg; diverged {100*pick['diverged_frac']:.0f}%")
        print(f"  tau_dist/kp floor at this gain: {pick['tau_over_kp_deg']:.3f} deg")
        if pick["tau_over_kp_deg"] > pick["median_deg"] * 3:
            print("  NOTE: the disturbance floor exceeds the achieved median -- feedforward is")
            print("        carrying the disturbance, so the floor is not binding here.")
    else:
        print(f"NO gain reaches mean alpha > {ALPHA_TARGET}. The allocator saturates across")
        print("the whole sweep, which would itself be the result: on this bus the LP is")
        print("starved at any gain stiff enough to reject the disturbance.")
    print("=" * 104)

    with open(f"{OUT}/select_gains_3plus1_{ts}.json", "w") as f:
        json.dump({"task": "select_gains_3plus1", "timestamp": ts, "n": N,
                   "kps": list(KPS), "alpha_target": ALPHA_TARGET,
                   "wheel": list(WHEEL), "h_frac": H_FRAC,
                   "tau_dist_Nm": TAU_DIST, "results": res}, f, indent=2)
    print(f"\nwrote {OUT}/select_gains_3plus1_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

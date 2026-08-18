"""Measure the planner's per-trial cost on the settled bus, so A's planner half is sized
from real numbers rather than a guessed multiplier.

Pre-committed cut order if the measured cost is bad (decided before the number existed, so
it is a pre-commitment rather than a deadline scramble):
  1. context-cell planner runs (3+0 and 3+3) -- least load-bearing: the planner paper
     already covers 3+0, and 3+3 planner adds nothing to this paper's argument;
  2. trial counts on those context cells;
  3. never the two 3+1 cells. The paper survives with planner data on 3+1 only.

Runs nice-d and with 2 workers so it does not distort Campaign A's timing on the same box.

Run: ``python papers/IAC_1RW/probe_planner_cost.py``
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from papers.IAC_1RW._iac_sim import T_ORBIT, error_series, make_config, simulate
from papers.IAC_1RW.generate_A_baseline import make_planner

OUT = os.path.join(os.path.dirname(__file__), "output_data")
N_PROBE = int(os.environ.get("PROBE_N", "3"))


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    print(f"planner cost probe: {N_PROBE} trials, 3+1 reduced, one orbit, settled bus")
    for rid in range(N_PROBE):
        c = make_config(rid, n_rw=1, task="reduced", tf=T_ORBIT, dt=1.0, seed=rid)
        c["h0"] = np.full(1, 0.05 * 15.0e-3)
        t0 = time.time()
        try:
            r = simulate(c, lambda sat, cfg: make_planner(sat, cfg),
                         disturbances=("gg", "drag", "srp", "dipole", "general"),
                         bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
            wall = time.time() - t0
            e = error_series(r)
            rows.append({"rid": rid, "wall_s": wall, "final_deg": float(e[-1]),
                         "ok": True})
            print(f"  trial {rid}: {wall/60:.1f} min, final {e[-1]:.3f} deg", flush=True)
        except Exception as exc:
            wall = time.time() - t0
            rows.append({"rid": rid, "wall_s": wall, "ok": False,
                         "error": f"{type(exc).__name__}"})
            print(f"  trial {rid}: FAILED after {wall/60:.1f} min "
                  f"({type(exc).__name__})", flush=True)

    ok = [r for r in rows if r.get("ok")]
    if ok:
        med = float(np.median([r["wall_s"] for r in ok]))
        print(f"\nmedian planner trial: {med/60:.1f} min "
              f"(PD trial for comparison: ~7-10 min under load)")
        for label, n_31, n_ctx in (("full plan (100/100/30/30)", 200, 120),
                                   ("3+1 only, 100 each", 200, 0),
                                   ("3+1 only, 50 each", 100, 0)):
            total = (n_31 + n_ctx) * med / 3600.0
            print(f"  {label:<28} -> {total:6.1f} h on one core-equivalent, "
                  f"~{total/10:5.1f} h on 10 workers")
    with open(f"{OUT}/planner_cost_{ts}.json", "w") as f:
        json.dump({"task": "planner_cost_probe", "timestamp": ts, "rows": rows}, f,
                  indent=2)
    print(f"\nwrote {OUT}/planner_cost_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

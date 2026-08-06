"""Bus-configuration ablation: star-tracker count x wheel size x actuator complement.

Three questions at once, all opened by the estimator fixes:

* **How many star trackers?** One is far easier to justify in a paper. The Earth keep-out at
  400 km is a 95.2 deg cone -- larger than a hemisphere -- so no single axis exceeds 45.5%
  availability, while an opposed pair reaches 90.9%. With the upgraded gyro the resulting
  outage costs 0.387 deg (one) against 0.098 deg (two), so one *may* now be enough.
* **Is the bigger wheel needed?** The wheel was upgraded from 2 mN m / 15 mN m s to
  7 / 50 at the same time as three genuine faults were fixed (gyro-bias Q, tracker mounting,
  damping). If the original wheel performs once those are corrected, keep it -- it is the
  smaller, cheaper, more conservative claim, and an upgrade that is not load-bearing should
  not be in the paper.
* **Does 3RW clear the pass criterion?** Median below ~0.05 deg with control error an order
  below knowledge error. 3RW is the control cell against which "3+1 approaches 3+3" is
  measured, so its number is load-bearing.

Run: ``python papers/IAC_1RW/ablate_bus_config.py``
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ADCS.controller import MTQ_w_RW_LP
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from papers.IAC_1RW._iac_sim import (BASELINE_H_FRAC, T_ORBIT, error_series,
                                     make_config, simulate)

OUT = os.path.join(os.path.dirname(__file__), "output_data")
N = int(os.environ.get("ABL_N", "6"))
J_TRANS = 0.13
KD_CRIT = 2.0 * np.sqrt(5e-5 * J_TRANS / 2.0)      # critical damping for THIS bus

WHEELS = {"orig 2/15": (2.0e-3, 15.0e-3), "big 7/50": (7.0e-3, 50.0e-3)}
TRACKERS = {"1 st": [np.array([1.0, 0.0, 0.0])],
            "2 st": [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]}


def _worker(cfg):
    tau_w, h_max = cfg["_wheel"]
    def mk(sat, config):
        return MTQ_w_RW_LP(est_sat=sat, p_gain=5e-5, d_gain=KD_CRIT, c_gain=1e-3,
                           h_target=config["h0"][0] * np.array([0.0, 0.0, 1.0]))
    return simulate(cfg, mk, bus_kwargs={"st_axes": cfg["_st"],
                                         "tau_w": tau_w, "h_max": h_max})


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S"); os.makedirs(OUT, exist_ok=True)
    print("=" * 100)
    print(f"Bus ablation -- tracker count x wheel x complement, n={N}, one orbit, "
          f"crit damping (kd={KD_CRIT:.2e})")
    print("PASS: 3RW median < ~0.05 deg, control error an order below knowledge error")
    print("=" * 100)
    print(f"\n{'complement':<8}{'wheel':<12}{'trk':<7}{'avail':>8}{'ecl':>7}"
          f"{'KNOWLEDGE':>12}{'POINTING':>11}{'p95 pt':>10}{'h end':>8}")
    print("-" * 100)
    res = {}
    for n_rw in (3, 1):
        for wname, (tau_w, h_max) in WHEELS.items():
            for tname, axes in TRACKERS.items():
                cfgs = []
                for rid in range(N):
                    c = make_config(rid, n_rw=n_rw, task="reduced", tf=T_ORBIT,
                                    dt=1.0, seed=rid)
                    # h0 must follow THIS config's h_max, not the module default.
                    c["h0"] = np.full(n_rw, BASELINE_H_FRAC * h_max)
                    c["_wheel"] = (tau_w, h_max); c["_st"] = axes
                    cfgs.append(c)
                runs = [r for r in MonteCarloRunner(
                    sim_func=_worker, config_generator=lambda i, _c=cfgs: _c[i],
                    num_runs=len(cfgs)).run() if r is not None]
                ks, ps, p95, av, ec, he = [], [], [], [], [], []
                for r in runs:
                    e = error_series(r); h = len(e) // 2
                    att = np.rad2deg(2*np.arccos(np.clip(np.abs(
                        np.sum(r["est"][:, 3:7]*r["state"][:, 3:7], axis=1)), 0, 1)))
                    ks.append(np.nanmedian(att[h:])); ps.append(np.median(e[h:]))
                    p95.append(np.percentile(e[h:], 95))
                    av.append(r["tracker_available"].mean()); ec.append(r["eclipse"].mean())
                    he.append(np.abs(r["state"][-1, 7:7+n_rw]).max()/h_max)
                key = f"{n_rw}rw|{wname}|{tname}"
                res[key] = {"n_rw": n_rw, "wheel": wname, "trackers": tname,
                            "median_knowledge_deg": float(np.median(ks)),
                            "median_pointing_deg": float(np.median(ps)),
                            "p95_pointing_deg": float(np.median(p95)),
                            "tracker_avail": float(np.mean(av)),
                            "eclipse_frac": float(np.mean(ec)),
                            "h_end_frac": float(np.median(he)),
                            "per_trial_knowledge": ks, "per_trial_pointing": ps,
                            "per_trial_avail": av, "per_trial_eclipse": ec}
                r_ = res[key]
                print(f"{str(n_rw)+'RW':<8}{wname:<12}{tname:<7}{r_['tracker_avail']:>8.3f}"
                      f"{r_['eclipse_frac']:>7.2f}{r_['median_knowledge_deg']:>12.4f}"
                      f"{r_['median_pointing_deg']:>11.4f}{r_['p95_pointing_deg']:>10.4f}"
                      f"{r_['h_end_frac']:>8.3f}", flush=True)
    with open(f"{OUT}/ablate_bus_{ts}.json", "w") as f:
        json.dump({"task": "ablate_bus_config", "timestamp": ts, "n": N,
                   "kd_crit": KD_CRIT, "results": res}, f, indent=2)
    print("=" * 100); print(f"\nwrote {OUT}/ablate_bus_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

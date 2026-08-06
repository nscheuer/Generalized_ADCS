"""What limits 3MTQ+1RW? Separate estimator error from control tuning from actuation.

Campaign A's reference cell (3+1, reduced attitude, one orbit) reads 53% at 5 deg but only
**2% at 1 deg**, median 4.81 deg. Something floors it well above the hardware's capability, and
the candidates need separating before anyone tunes anything:

* **Estimator error.** You cannot point better than you know your attitude. An earlier probe
  showed ~5.9 deg median estimated-attitude error, which would by itself explain a 4.8 deg
  pointing median and a 2% figure at 1 deg.
* **Control tuning.** The gains are inherited from genACS, whose bus is 1.2 kg with
  ``J = diag(0.022, 0.022, 0.004)``. This bus is 12 kg with ``J = diag(0.13, 0.10, 0.05)`` --
  roughly **6x the inertia**. For PD on a rigid body the closed loop is
  ``J theta_ddot + kd theta_dot + (kp/2) theta = 0``, so critical damping wants
  ``kd = 2 sqrt(kp J / 2)``: at kp = 5e-5 and J = 0.13 that is ~3.7e-3, against the 1e-3
  actually in use. The loop is under-damped by ~4x and should ring rather than settle.
* **Actuation.** Peak wheel momentum reached only 0.237 of h_max and magnetorquer duty settled
  at 0.454, so neither is saturated -- actuation is unlikely to be the binding constraint, but
  the truth-state cells below confirm it rather than assuming it.
* **Update rate.** Cheap to test alongside.

Conditions are paired by seed, so every difference is the condition and not the draw.

Run: ``python papers/IAC_1RW/diagnose_3plus1_limit.py``
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
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from ADCS.satellite_factory import IAC_6U
from papers.IAC_1RW._iac_sim import T_ORBIT, error_series, make_config, simulate

OUT = os.path.join(os.path.dirname(__file__), "output_data")

J_BORE = 0.05          # inertia about the boresight [kg m^2]
J_TRANS = 0.13         # largest transverse inertia

#: (label, kp, kd, kc). "crit" is critical damping for THIS bus at the genACS kp.
GAINSETS = {
    "genACS":     (5e-5, 1e-3, 1e-3),
    "crit":       (5e-5, 2.0 * np.sqrt(5e-5 * J_TRANS / 2.0), 1e-3),
    "stiff-crit": (2e-4, 2.0 * np.sqrt(2e-4 * J_TRANS / 2.0), 1e-3),
}

N_TRIALS = int(os.environ.get("DIAG_N", "16"))
TF = float(os.environ.get("DIAG_TF", str(T_ORBIT)))


def _worker(cfg):
    kp, kd, kc = cfg["gains"]

    def mk(sat, config):
        return MTQ_w_RW_LP(est_sat=sat, p_gain=kp, d_gain=kd, c_gain=kc,
                           h_target=np.zeros(3))

    return simulate(cfg, mk, use_estimator=cfg["use_estimator"])


def summarise(runs: List[Dict[str, Any]], hold_frac: float = 0.25) -> Dict[str, Any]:
    fin, held, est_err = [], [], []
    for r in runs:
        if r is None:
            continue
        e = error_series(r)
        k0 = int((1.0 - hold_frac) * e.size)
        fin.append(float(e[-1]))
        held.append(float(np.percentile(e[k0:], 95)))
        if np.isfinite(r["est"]).any():
            q_t = r["state"][k0:, 3:7]
            q_e = r["est"][k0:, 3:7]
            good = np.isfinite(q_e).all(axis=1)
            if good.any():
                dots = np.abs(np.sum(q_t[good] * q_e[good], axis=1))
                est_err.append(float(np.median(
                    np.rad2deg(2.0 * np.arccos(np.clip(dots, 0, 1))))))
    fin = np.asarray(fin)
    return {
        "n": int(fin.size),
        "conv5": float(100 * np.mean(fin < 5.0)),
        "conv1": float(100 * np.mean(fin < 1.0)),
        "conv0p1": float(100 * np.mean(fin < 0.1)),
        "median_final_deg": float(np.median(fin)),
        "median_held_p95_deg": float(np.median(held)) if held else float("nan"),
        "median_est_att_err_deg": float(np.median(est_err)) if est_err else None,
        "finals": fin.tolist(),
    }


def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    conditions = [
        ("A baseline (est, genACS gains)",      True,  "genACS", 1.0),
        ("B truth state (genACS gains)",        False, "genACS", 1.0),
        ("C est + critical damping",            True,  "crit",   1.0),
        ("D truth + critical damping",          False, "crit",   1.0),
        ("E truth + stiffer critical",          False, "stiff-crit", 1.0),
        ("F est + crit + dt=0.5 s",             True,  "crit",   0.5),
    ]

    print("=" * 96)
    print(f"What limits 3MTQ+1RW?  n={N_TRIALS} paired trials, reduced attitude, "
          f"tf={TF:.0f} s")
    for lbl, (kp, kd, kc) in GAINSETS.items():
        zeta = kd / (2.0 * np.sqrt(kp * J_TRANS / 2.0)) if kp > 0 else float("nan")
        print(f"  gains {lbl:<11} kp={kp:.1e} kd={kd:.2e} -> damping ratio "
              f"zeta = {zeta:.2f} (J={J_TRANS})")
    print("=" * 96)

    results: Dict[str, Any] = {}
    print(f"\n{'condition':<34}{'conv5':>8}{'conv1':>8}{'conv0.1':>9}"
          f"{'median':>10}{'held p95':>10}{'est err':>10}")
    print("-" * 96)
    for label, use_est, gkey, dtf in conditions:
        cfgs = []
        for rid in range(N_TRIALS):
            c = make_config(rid, n_rw=1, task="reduced", tf=TF, dt=1.0 * dtf, seed=rid)
            c["gains"] = GAINSETS[gkey]
            c["use_estimator"] = use_est
            cfgs.append(c)
        runner = MonteCarloRunner(sim_func=_worker,
                                  config_generator=lambda i, _c=cfgs: _c[i],
                                  num_runs=len(cfgs))
        runs = [r for r in runner.run() if r is not None]
        m = summarise(runs)
        results[label] = m
        ee = ("-" if m["median_est_att_err_deg"] is None
              else f"{m['median_est_att_err_deg']:.3f}")
        print(f"{label:<34}{m['conv5']:>7.1f}%{m['conv1']:>7.1f}%{m['conv0p1']:>8.1f}%"
              f"{m['median_final_deg']:>10.3f}{m['median_held_p95_deg']:>10.3f}{ee:>10}")

    # ---- attribution -------------------------------------------------------------------
    print("\n" + "=" * 96)
    a = results["A baseline (est, genACS gains)"]
    b = results["B truth state (genACS gains)"]
    c = results["C est + critical damping"]
    d = results["D truth + critical damping"]
    print("Attribution (median final pointing error):")
    print(f"  baseline                         {a['median_final_deg']:8.3f} deg")
    print(f"  remove estimator error only      {b['median_final_deg']:8.3f} deg  "
          f"(delta {a['median_final_deg']-b['median_final_deg']:+.3f})")
    print(f"  fix damping only                 {c['median_final_deg']:8.3f} deg  "
          f"(delta {a['median_final_deg']-c['median_final_deg']:+.3f})")
    print(f"  both                             {d['median_final_deg']:8.3f} deg  "
          f"(delta {a['median_final_deg']-d['median_final_deg']:+.3f})")
    if a["median_est_att_err_deg"]:
        print(f"\n  estimated-attitude error in the held interval: "
              f"{a['median_est_att_err_deg']:.3f} deg -- pointing cannot beat this.")
    print("=" * 96)

    with open(f"{OUT}/diag_3plus1_{ts}.json", "w") as f:
        json.dump({"task": "diagnose_3plus1", "timestamp": ts, "n": N_TRIALS,
                   "tf_s": TF, "gainsets": {k: list(v) for k, v in GAINSETS.items()},
                   "results": results}, f, indent=2)
    print(f"\nwrote {OUT}/diag_3plus1_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

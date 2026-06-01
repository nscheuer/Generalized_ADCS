"""P2.4 tuning pass -- does disabling the goal-blend / tuning control_reg close
the MPC-vs-TVLQR gap?

Paired campaigns (same seeds -> same planned trajectories) on 3+1 / reduced:
  * TVLQR
  * MPC default        (goal-blend ON,  control_reg=1e-3)  -- the committed cfg
  * MPC blend-off      (pure tracking,  control_reg=1e-3)
  * MPC blend-off,reg2 (pure tracking,  control_reg=1e-2)

Reduced trial count (P24T_TRIALS, default 25) for a quick read; if a variant
clearly wins, re-run the full 100-trial P2.4 with that config. Emits
output_data/P2.4_tuning_<ts>.json.
"""

import os
import sys
import json
import datetime as _dt

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
import _paper2_sim as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
CONFIG, BASE_SEED, CONV = "3+1", 42, 5.0


def mc_config():
    return ADCS.MCConfig(
        w=lambda rng: ADCS.helpers.normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
        h=lambda rng: rng.uniform(-1e-4, 1e-4, size=1),
        goal=lambda rng: ADCS.goals.ECI_Goal(
            eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))),
        orbit=P.make_random_os)


def run(factory, n):
    sat = P.make_sat(CONFIG, estimated=False)
    return ADCS.simulate_mc(
        x=P.x0(1), satellite=sat, controller=factory(sat),
        goal=ADCS.goals.ECI_Goal(np.array([1.0, 0, 0])), os0=P.default_os0(),
        dt=1.0, tf=1000.0, mc_config=mc_config(), num_runs=n, base_seed=BASE_SEED)


def metrics(res):
    try:
        bu = res.satellite.get_boresight(None)
    except Exception:
        bu = res.satellite.boresight
        bu = bu.get("default", next(iter(bu.values()))) if isinstance(bu, dict) else bu
    bu = np.asarray(bu, float).reshape(3); bu /= np.linalg.norm(bu)
    finals, effort = [], []
    for run_ in res.runs:
        st = np.asarray(run_.state_hist, float); tg = np.asarray(run_.target_hist, float)
        finals.append(_angle_deg(_boresight_eci(st[-1, 3:7], bu), tg[-1][1:4]))
        u = getattr(run_, "control_hist", None)
        effort.append(float(np.sum(np.linalg.norm(np.asarray(u, float), axis=1)))
                      if u is not None else float("nan"))
    finals = np.asarray(finals)
    return {"n": int(finals.size), "conv_pct": float(100 * np.mean(finals < CONV)),
            "mean_final_deg": float(np.mean(finals)),
            "p95_final_deg": float(np.percentile(finals, 95)),
            "mean_effort": float(np.nanmean(effort))}


def main():
    ts = os.environ.get("P24T_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    n = int(os.environ.get("P24T_TRIALS", "25"))
    ps = lambda s: P.make_planner_settings(s)
    variants = {
        "tvlqr": lambda s: ADCS.controller.Plan_and_Track_LQR(est_sat=s, planner_settings=ps(s)),
        "mpc_default": lambda s: ADCS.controller.Plan_and_Track_SingleStepMPC(
            est_sat=s, planner_settings=ps(s)),
        "mpc_blendoff": lambda s: ADCS.controller.Plan_and_Track_SingleStepMPC(
            est_sat=s, planner_settings=ps(s), disable_goal_blend=True),
        "mpc_blendoff_reg1e-2": lambda s: ADCS.controller.Plan_and_Track_SingleStepMPC(
            est_sat=s, planner_settings=ps(s), disable_goal_blend=True, control_reg=1e-2),
    }
    print(f"[P2.4-tune] n={n} ts={ts}")
    out = {}
    for name, fac in variants.items():
        print(f"  running {name} ...")
        m = metrics(run(fac, n))
        out[name] = m
        print(f"    {name:22s} conv {m['conv_pct']:5.1f}%  mean {m['mean_final_deg']:7.2f}  "
              f"p95 {m['p95_final_deg']:7.2f}  effort {m['mean_effort']:.1f}")
    payload = {"task": "P2.4_tuning", "timestamp": ts, "n_trials": n,
               "config": CONFIG, "variants": out}
    with open(f"{OUT}/P2.4_tuning_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P2.4-tune] wrote {OUT}/P2.4_tuning_{ts}.json")
    return payload


if __name__ == "__main__":
    main()

"""P2.6 -- Replan after disturbance (Paper 2, online replanning demo).

The planner produces an open-loop trajectory; closed-loop TVLQR tracks it; an
unexpected impulsive attitude disturbance perturbs the state mid-maneuver; the
planner re-solves from the perturbed state and tracking resumes on the new
trajectory. A single clean illustrative trajectory (not an MC).

Scenario:
  * Config 3+1 (consistent with P2.1/P2.2).
  * Goal: anti-ram (AntiVelocity_Goal) pointing acquisition.
  * Initial: at rest, boresight offset from the anti-ram target.
  * First plan over ~200 s; at t = 100 s inject an impulsive +30 deg attitude
    perturbation about a fixed axis.
  * Replan trigger: state-based (pointing error exceeds a threshold) -- the
    perturbation trips it immediately; replan from the perturbed state.

Three traces on pointing-error vs time:
  1. original open-loop plan (what would have happened, no disturbance),
  2. disturbed + NO replan (track the stale plan through the kick),
  3. disturbed + replan (re-solve and recover).

Outputs:
  output_data/P2.6_replan_<ts>.json
  output_data/fig_replan.{png,pdf}
  P2.6_RESULTS.md (written separately)
"""

import os
import sys
import json
import time
import datetime as _dt

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.math_helpers import normalize, rot_mat, quat_mult, rot_exp
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.orbit import Orbit
import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
os.makedirs(OUT, exist_ok=True)

CONFIG = "3+1"
TF = 1000.0                 # gentle horizon: realized TVLQR tracks the plan
DT = 1.0
T_DISTURB = 500.0           # inject after acquisition has settled
DIST_ANGLE_DEG = 30.0
DIST_AXIS = normalize(np.array([1.0, -0.5, 0.3]))
REPLAN_ERR_DEG = 10.0       # state-based replan trigger
CONV_DEG = 5.0
SEC2CENT = TimeConstants.sec2cent


def perturb_quat(q, axis, angle_rad):
    """Apply an impulsive body rotation: q_new = q (x) dq(axis, angle)."""
    dq = np.concatenate(
        [[np.cos(angle_rad / 2)], np.sin(angle_rad / 2) * normalize(axis)])
    return normalize(quat_mult(q, dq))


def bore_unit(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        if isinstance(b, dict):
            b = b.get("default", next(iter(b.values())))
    return normalize(np.asarray(b, float).reshape(3))


def run_episode(replan: bool):
    """Single trajectory. Returns dicts of time series + diagnostics.

    If ``replan`` is True, a state-based trigger re-solves the trajectory from
    the perturbed state once the tracking error exceeds REPLAN_ERR_DEG after
    the disturbance.
    """
    sat = P.make_sat(CONFIG, estimated=False)
    bu = bore_unit(sat)
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=sat, planner_settings=P.make_planner_settings(sat))

    os0 = P.default_os0()
    orb = Orbit(os0=os0, end_time=os0.J2000 + (TF + 5) * SEC2CENT, dt=DT,
                use_J2=True, fast=False, verbose=False)
    N = int(TF / DT)
    os_seq = [orb.get_os(J2000=os0.J2000 + k * DT * SEC2CENT) for k in range(N + 1)]

    goal = ADCS.goals.AntiVelocity_Goal()
    gl = ADCS.GoalList(goal_timeline={0.0: goal}, time_units="seconds",
                       start_juliantime=os0.J2000)

    # initial state: at rest, identity attitude
    x = P.x0(1)

    # --- first plan (acquisition) ---
    traj1 = ctrl.calculate_trajectory(os0.J2000, TF, x, os0, gl)
    ctrl.set_active_trajectory(traj1)

    # original open-loop planned boresight error vs target (no disturbance)
    plan_t = np.asarray(traj1.times, float)
    plan_err = np.array([
        _angle_deg(_boresight_eci(traj1.get_state_at(tt)[3:7], bu),
                   gl.get_active_goal(tt, time_units="centuries").to_ref(
                       orb.get_os(J2000=tt))[0][1:4])
        for tt in plan_t])
    plan_t_s = (plan_t - plan_t[0]) / SEC2CENT

    t_hist, err_hist, u_norm_hist = [], [], []
    replan_time = None
    replan_solve_ms = None
    disturbed = False
    for k in range(N):
        os_k = os_seq[k]
        ct = os_k.J2000
        ag = gl.get_active_goal(ct, time_units="centuries")
        target = ag.to_ref(os_k)[0][1:4]
        err = _angle_deg(_boresight_eci(x[3:7], bu), target)
        t_s = k * DT
        t_hist.append(t_s); err_hist.append(err)

        # inject disturbance once
        if not disturbed and t_s >= T_DISTURB:
            x[3:7] = perturb_quat(x[3:7], DIST_AXIS, np.radians(DIST_ANGLE_DEG))
            disturbed = True

        # state-based replan trigger (after disturbance)
        if replan and disturbed and replan_time is None and err > REPLAN_ERR_DEG \
                and t_s > T_DISTURB:
            t0 = time.perf_counter()
            traj2 = ctrl.calculate_trajectory(ct, TF - t_s, x.copy(), os_k, gl)
            replan_solve_ms = (time.perf_counter() - t0) * 1e3
            ctrl.set_active_trajectory(traj2)
            replan_time = t_s

        u = ctrl.find_u(x_hat=x, sens=sat.sensor_readings(x=x, os=os_k),
                        est_sat=sat, os_hat=os_k, goal=ag)
        u_norm_hist.append(float(np.linalg.norm(u)))

        os_next = os_seq[k + 1]
        out = solve_ivp(fun=sat.dynamics_for_solver, t_span=(0, DT), y0=x,
                        method="RK45", args=(u, os_k, os_next),
                        rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]; x[3:7] = normalize(x[3:7])

    return {
        "t": np.asarray(t_hist), "err": np.asarray(err_hist),
        "u_norm": np.asarray(u_norm_hist),
        "plan_t": plan_t_s, "plan_err": plan_err,
        "replan_time": replan_time, "replan_solve_ms": replan_solve_ms,
    }


def recovery_time(t, err, t_replan):
    """Time after replan until error stays < CONV_DEG."""
    if t_replan is None:
        return float("nan")
    m = t >= t_replan
    tt, ee = t[m], err[m]
    above = ee > CONV_DEG
    if not above.any():
        return 0.0
    if above[-1]:
        return float("nan")
    return float(tt[int(np.flatnonzero(above)[-1]) + 1] - t_replan)


def main():
    ts = os.environ.get("P26_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P2.6] config={CONFIG} disturb={DIST_ANGLE_DEG}deg@{T_DISTURB}s ts={ts}")

    print("[P2.6] episode: disturbed + replan ...")
    rp = run_episode(replan=True)
    print("[P2.6] episode: disturbed + NO replan ...")
    nr = run_episode(replan=False)

    err_at_trigger = float(rp["err"][np.argmin(np.abs(rp["t"] - (rp["replan_time"] or T_DISTURB)))]) \
        if rp["replan_time"] is not None else float("nan")
    rec_t = recovery_time(rp["t"], rp["err"], rp["replan_time"])
    eff_replan = float(np.sum(rp["u_norm"]) * DT)
    eff_noreplan = float(np.sum(nr["u_norm"]) * DT)

    # figure
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rp["plan_t"], rp["plan_err"], color="C2", lw=1.4, ls="-",
            label="original open-loop plan (no disturbance)")
    ax.plot(nr["t"], nr["err"], color="C3", lw=1.5, ls="--",
            label="disturbed, NO replan (stale plan)")
    ax.plot(rp["t"], rp["err"], color="C0", lw=1.8,
            label="disturbed + replan (recovery)")
    ax.axvline(T_DISTURB, color="k", ls=":", lw=1.0)
    ax.annotate("disturbance\n(+30 deg)", xy=(T_DISTURB, ax.get_ylim()[1] * 0.5),
                fontsize=8, ha="left")
    if rp["replan_time"] is not None:
        ax.axvline(rp["replan_time"], color="C0", ls=":", lw=1.0)
        ax.annotate("replan", xy=(rp["replan_time"], 1.0), fontsize=8,
                    color="C0", ha="left")
    ax.axhline(CONV_DEG, color="gray", ls=":", lw=0.7)
    ax.set_yscale("log"); ax.set_xlabel("time [s]")
    ax.set_ylabel("pointing error [deg]")
    ax.set_title("P2.6: online replan after disturbance (3+1, anti-ram)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_replan.png", dpi=150); fig.savefig(f"{OUT}/fig_replan.pdf")
    plt.close(fig)

    payload = {
        "task": "P2.6_replan", "timestamp": ts, "config": CONFIG,
        "scenario": {"tf": TF, "dt": DT, "t_disturb": T_DISTURB,
                     "disturb_angle_deg": DIST_ANGLE_DEG,
                     "disturb_axis": DIST_AXIS.tolist(),
                     "replan_trigger": f"error > {REPLAN_ERR_DEG} deg after disturbance",
                     "goal": "AntiVelocity_Goal"},
        "results": {
            "replan_time_s": rp["replan_time"],
            "error_at_replan_deg": err_at_trigger,
            "recovery_time_s": rec_t,
            "replan_solve_ms": rp["replan_solve_ms"],
            "control_effort_replan": eff_replan,
            "control_effort_no_replan": eff_noreplan,
            "final_error_replan_deg": float(rp["err"][-1]),
            "final_error_no_replan_deg": float(nr["err"][-1]),
        },
        "series": {
            "t": rp["t"].tolist(), "err_replan": rp["err"].tolist(),
            "err_no_replan": nr["err"].tolist(),
            "plan_t": rp["plan_t"].tolist(), "plan_err": rp["plan_err"].tolist(),
        },
    }
    with open(f"{OUT}/P2.6_replan_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    r = payload["results"]
    print(f"[P2.6] replan@{r['replan_time_s']}s err_at_replan={r['error_at_replan_deg']:.1f}deg "
          f"recovery={r['recovery_time_s']}s solve={r['replan_solve_ms']:.1f}ms")
    print(f"[P2.6] final err: replan={r['final_error_replan_deg']:.2f} "
          f"no_replan={r['final_error_no_replan_deg']:.2f}; "
          f"effort replan={r['control_effort_replan']:.1f} no_replan={r['control_effort_no_replan']:.1f}")
    print(f"[P2.6] wrote {OUT}/P2.6_replan_{ts}.json + fig_replan")
    return payload


if __name__ == "__main__":
    main()

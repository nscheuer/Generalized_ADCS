"""Campaign B (planner) -- equal-peak-torque ablation, and the floor-vs-bracket diagnostic.

**Why this replaces the PD version.** The exponent claim is a statement about minimum-time
optimal control: it says a bracket manoeuvre *exists* reaching the along-field direction in
``Theta^(1/3)``. Direction-preserving LP allocation cannot execute one. When the requested
torque lies along **B** the LP scales it to zero by construction -- the same all-or-nothing
property genACS documents, and the reason LP beats QP on 3+1. So on a magnetorquer-only bus the
along-field error component receives no command at all, at any gain and at any dipole; the
controller simply waits for the field to rotate.

Measured, before this rewrite: at equal peak torque the PD/LP 3MTQ+0RW bus completed **zero**
rest-to-rest slews inside four orbits at Theta = 0.05, 0.1 and 0.2 rad -- including a
cross-field axis. That is not a null result about geometry. It is the allocator behaving
exactly as designed, and it is a Section V finding in its own right: the theoretical agility of
a magnetorquer-only bus is structurally inaccessible to direction-preserving feedback, so part
of the wheel's practical value is that it removes the need for manoeuvres feedback cannot
perform.

**The diagnostic this run settles.** Two candidate models for third-order authority:

* **Bracket.** The rotating field supplies the bracket, third-order authority goes as
  ``tau_max * omega_B``, and completion time falls as torque rises -- scaling as
  ``Theta^(1/3)``.
* **Floor.** The along-field direction is gated by the field rotating in the body frame, so
  there is a floor at some fraction of an orbit that no amount of torque removes.

They are distinguished by sweeping ``Theta`` **and** ``m_max`` together:

===========================================  ==========================================
observation                                  conclusion
===========================================  ==========================================
t_f falls with m_max, scales as Theta^(1/3)  bracket; the exponent claim stands
t_f pinned near a fixed fraction of an orbit floor; no torque removes it
  regardless of Theta and m_max
===========================================  ==========================================

The floor answer is qualitatively different and strictly better for the paper: "the wheel
removes an orbit-timescale floor" is a stronger claim than "the wheel improves an exponent".

Planner recipe (do not simplify -- each item was learned the hard way):
  * plan past the executed window, execute only the first part;
  * cap the AL/iLQR iteration counts;
  * ``trajOpt`` **raises** on non-convergence, so every call is wrapped and fallbacks counted;
  * a wall-clock backstop, because rare drifted states grind for minutes.

Run: ``python papers/IAC_1RW/generate_B_equal_torque_planner.py``  (``B_SCALE=fast`` to smoke)
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.helpers.math_helpers import normalize, quat_mult, rot_mat
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory import create_iac_6u_bus, IAC_6U

from papers.IAC_1RW._iac_sim import EPOCH, T_ORBIT, _get_orbit
from papers.IAC_1RW.generate_B_equal_torque import (
    ALIGN_HI, ALIGN_LO, fit_slope, field_samples, solve_equal_torque_m_max,
)

OUT = os.path.join(os.path.dirname(__file__), "output_data")

THETAS_RAD = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
#: Torque scalings applied to the equal-peak-torque dipole. This axis is the floor test.
M_SCALES = (0.5, 1.0, 2.0, 4.0)
N_AXES = 4

DONE_ANGLE_RAD = np.deg2rad(2.0)
DONE_RATE_RPS = np.deg2rad(0.02)

#: Two orbits. A floor, if there is one, lives at a fraction of an orbit; a bracket that needs
#: more than two orbits is not an agility story anyone will care about.
T_MAX_S = 2.0 * T_ORBIT
PLAN_WINDOW_S = 500.0
PLAN_OVERLAP_S = 500.0
PLAN_TIMEOUT_S = 90.0

SCALES = {
    "fast":  {"thetas": (0.1, 0.5), "m_scales": (1.0,), "n_axes": 1,
              "t_max": 0.5 * T_ORBIT},
    "paper": {"thetas": THETAS_RAD, "m_scales": M_SCALES, "n_axes": N_AXES,
              "t_max": T_MAX_S},
}


def scale():
    return SCALES[os.environ.get("B_SCALE", "paper")]


class _Timeout(Exception):
    pass


def _with_timeout(seconds, fn, *a, **kw):
    """Wall-clock backstop. Rare drifted states make the planner grind for minutes."""
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def make_planner(sat):
    from ADCS.controller import Plan_and_Track_LQR
    from ADCS.controller.plan_and_track import CostWeights, PlannerSettings

    ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)
    ps.verbosity = False
    ps.cost_main.use_full_cost_hessian = True
    ps.pass1.regularization.use_dynamics_hess = 1
    ps.init_traj.bdot_gain = 500
    ps.pass1.aug_lag.penalty_init = 1e-3
    ps.pass1.aug_lag.penalty_scale = 10
    ps.pass1.convergence.max_outer_iter = 10
    ps.pass1.convergence.max_inner_iter = 25
    ps.pass2.aug_lag.penalty_init = 1e5
    ps.pass2.aug_lag.penalty_scale = 10
    ps.pass2.convergence.max_outer_iter = 6
    ps.pass2.convergence.max_inner_iter = 15
    ps.cost_main = CostWeights(
        angle=1e1, angle_N=1e1, ang_vel=1e5, ang_vel_N=1e5,
        ang_vel_err_dir=1e2, ang_vel_err_dir_N=0.0, ang_vel_mag=0.0,
        ang_vel_mag_N=0.0, control_mult=1.0, ang_cost_func_type=2)
    ps.cost_second = ps.cost_main
    ps.cost_tvlqr = CostWeights(
        angle=1e5, angle_N=1e6, ang_vel=1e6, ang_vel_N=1e8,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0, control_mult=1.0,
        ang_cost_func_type=2)
    return Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)


def slew_config(run_id, *, bus, theta, axis_seed, m_max, m_scale, t_max):
    rng = np.random.default_rng(axis_seed)
    e = normalize(rng.standard_normal(3))
    q0 = normalize(rng.standard_normal(4))
    dq = np.concatenate([[np.cos(theta / 2.0)], e * np.sin(theta / 2.0)])
    return {"run_id": run_id, "bus": bus, "theta": float(theta),
            "axis_seed": axis_seed, "m_max": float(m_max), "m_scale": float(m_scale),
            "q0": q0, "q_goal": normalize(quat_mult(q0, dq)), "slew_axis_body": e,
            "raan_deg": float(rng.uniform(0, 360)), "phase_deg": float(rng.uniform(0, 360)),
            "inc_deg": 97.0, "n_rw": 1 if bus == "3+1" else 0,
            "tf": float(t_max), "dt": 1.0, "task": "full"}


def run_slew(config: Dict[str, Any]) -> Dict[str, Any]:
    slot = claim_worker_slot()
    try:
        n_rw, dt = config["n_rw"], config["dt"]
        t_max = config["tf"]

        kw = {"disturbances": (), "estimate_dipole": False}
        if n_rw == 0:
            kw["m_max"] = config["m_max"]
        sat = create_iac_6u_bus(n_rw=n_rw, **kw)

        orb = _get_orbit(config, dt, t_max + 2 * PLAN_OVERLAP_S)
        goal = Fixed_Attitude_Goal(np.asarray(config["q_goal"], float))
        planner = make_planner(sat)
        fb = MTQ_w_RW_LP(est_sat=sat, p_gain=5e-5, d_gain=1e-3, c_gain=1e-3,
                         h_target=np.zeros(3))

        x = np.concatenate([np.zeros(3), config["q0"], np.zeros(n_rw)])
        qr = normalize(np.asarray(config["q_goal"], float))

        os0 = orb.get_os(J2000=EPOCH)
        B0 = rot_mat(config["q0"]).T @ np.asarray(os0.B, float)
        align = float(abs(np.asarray(config["slew_axis_body"], float)
                          @ (B0 / np.linalg.norm(B0))))

        t, t_done, n_fallback, n_plans = 0.0, np.nan, 0, 0
        steps = int(t_max / dt)
        next_replan = 0.0
        use = fb

        for i in range(steps):
            if i % 200 == 0:
                update_worker_progress(slot, config["run_id"], i, steps)
            os_k = orb.get_os(J2000=EPOCH + t * TimeConstants.sec2cent)

            if t >= next_replan:
                gl = GoalList({os_k.J2000: goal})
                try:
                    traj = _with_timeout(PLAN_TIMEOUT_S, planner.calculate_trajectory,
                                         os_k.J2000, PLAN_WINDOW_S + PLAN_OVERLAP_S,
                                         x.copy(), os_k, gl)
                    planner.set_active_trajectory(traj)
                    use, n_plans = planner, n_plans + 1
                except Exception:
                    use, n_fallback = fb, n_fallback + 1
                next_replan = t + PLAN_WINDOW_S

            ang = 2.0 * np.arccos(np.clip(abs(float(x[3:7] @ qr)), 0.0, 1.0))
            if ang < DONE_ANGLE_RAD and np.linalg.norm(x[0:3]) < DONE_RATE_RPS:
                t_done = t
                break

            try:
                sens = sat.sensor_readings(x=x, os=os_k)
                u = np.asarray(use.find_u(x_hat=x, sens=sens, est_sat=sat,
                                          os_hat=os_k, goal=goal), float)
            except Exception:
                u = np.asarray(fb.find_u(x_hat=x, sens=sat.sensor_readings(x=x, os=os_k),
                                         est_sat=sat, os_hat=os_k, goal=goal), float)

            t += dt
            os_n = orb.get_os(J2000=EPOCH + t * TimeConstants.sec2cent)
            x = solve_ivp(sat.dynamics_for_solver, (0, dt), x, method="RK45",
                          args=(u, os_k, os_n), rtol=1e-6, atol=1e-6).y[:, -1]
            x[3:7] = normalize(x[3:7])

        return {"run_id": config["run_id"], "bus": config["bus"],
                "theta": config["theta"], "m_scale": config["m_scale"],
                "alignment": align, "t_done_s": float(t_done),
                "completed": bool(np.isfinite(t_done)),
                "t_done_orbits": float(t_done / T_ORBIT) if np.isfinite(t_done) else np.nan,
                "n_plans": n_plans, "n_fallback": n_fallback}
    except Exception as exc:
        return {"run_id": config["run_id"], "bus": config["bus"],
                "theta": config["theta"], "m_scale": config["m_scale"],
                "alignment": float("nan"), "t_done_s": float("nan"),
                "completed": False, "t_done_orbits": float("nan"),
                "n_plans": 0, "n_fallback": 0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        release_worker_slot(slot)


def main() -> int:
    s = scale()
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    Bs = field_samples(120)
    eq = solve_equal_torque_m_max(Bs, IAC_6U.tau_w)
    B_med = float(np.median(np.linalg.norm(Bs, axis=1)))

    print("=" * 96)
    print("Campaign B (PLANNER) -- equal-peak-torque ablation + floor-vs-bracket diagnostic")
    print("=" * 96)
    print(f"\nequal-peak-torque m_max = {eq['m_max']:.2f} A m^2 at a median field of "
          f"{B_med*1e6:.1f} uT")
    print(f"  (naive tau_w/|B| gives {eq['naive_m_max']:.2f}, "
          f"{eq['naive_m_max']/eq['m_max']:.2f}x too large)")
    print(f"  peak torque over the orbit: median {eq['median_peak_Nm']*1e3:.3f}, "
          f"min {eq['min_peak_Nm']*1e3:.3f}, max {eq['max_peak_Nm']*1e3:.3f} mN m")
    print("  PHYSICALLY ABSURD BY CONSTRUCTION -- an ablation, not a design proposal.")

    cfgs, rid = [], 0
    for bus in ("3+1", "3+0"):
        ms = (1.0,) if bus == "3+1" else s["m_scales"]
        for m_scale in ms:
            for th in s["thetas"]:
                for k in range(s["n_axes"]):
                    cfgs.append(slew_config(
                        rid, bus=bus, theta=th, axis_seed=200000 + 1000 * k + int(th * 100),
                        m_max=eq["m_max"] * m_scale, m_scale=m_scale, t_max=s["t_max"]))
                    rid += 1

    print(f"\nrunning {len(cfgs)} planner slews, horizon {s['t_max']/T_ORBIT:.1f} orbits...")
    runner = MonteCarloRunner(sim_func=run_slew,
                              config_generator=lambda i, _c=cfgs: _c[i],
                              num_runs=len(cfgs))
    res = [r for r in runner.run() if r is not None]

    # ---- floor vs bracket -------------------------------------------------------------
    print("\n" + "=" * 96)
    print("FLOOR-vs-BRACKET: does completion time fall as torque rises?")
    print(f"{'bus':<6}{'m_scale':>9}{'n done':>9}{'med t [orb]':>13}{'med t [s]':>12}"
          f"{'slope vs Theta':>16}")
    print("-" * 96)
    diag = {}
    for bus in ("3+1", "3+0"):
        for m_scale in ((1.0,) if bus == "3+1" else s["m_scales"]):
            sel = [r for r in res if r["bus"] == bus and r["m_scale"] == m_scale]
            done = [r for r in sel if r["completed"]]
            th = np.array([r["theta"] for r in sel])
            tt = np.array([r["t_done_s"] for r in sel])
            f = fit_slope(th, tt)
            med_o = float(np.nanmedian([r["t_done_orbits"] for r in done])) if done else np.nan
            med_s = float(np.nanmedian([r["t_done_s"] for r in done])) if done else np.nan
            diag[f"{bus}|{m_scale}"] = {"n_total": len(sel), "n_done": len(done),
                                        "median_orbits": med_o, "median_s": med_s,
                                        "fit": f}
            print(f"{bus:<6}{m_scale:>9.1f}{len(done):>4}/{len(sel):<4}"
                  f"{med_o:>13.3f}{med_s:>12.1f}{f['slope']:>16.3f}")

    # ---- field-classified slope fits ---------------------------------------------------
    print("\n" + "-" * 96)
    print("Slope by field alignment (equal peak torque, m_scale = 1.0)")
    print(f"{'bus':<6}{'family':<14}{'n':>5}{'slope':>9}{'r2':>8}   predicted")
    fams = {}
    for bus in ("3+1", "3+0"):
        for fam, lo, hi in (("along-field", ALIGN_HI, 1.01), ("cross-field", -0.01, ALIGN_LO)):
            sel = [r for r in res if r["bus"] == bus and r["m_scale"] == 1.0
                   and np.isfinite(r["alignment"]) and lo <= r["alignment"] < hi]
            f = fit_slope(np.array([r["theta"] for r in sel]),
                          np.array([r["t_done_s"] for r in sel]))
            fams[f"{bus}|{fam}"] = {"fit": f, "n_total": len(sel)}
            pred = ("~1/2" if bus == "3+1" else
                    ("~1/3 (weight-3 bracket)" if fam == "along-field" else "~1/2"))
            print(f"{bus:<6}{fam:<14}{f['n']:>5}{f['slope']:>9.3f}{f['r2']:>8.3f}   {pred}")

    nfb = sum(r.get("n_fallback", 0) for r in res)
    npl = sum(r.get("n_plans", 0) for r in res)
    print(f"\nplanner: {npl} successful plans, {nfb} fallbacks to reactive feedback")
    print("=" * 96)

    payload = {"task": "B_equal_torque_planner", "timestamp": ts,
               "equal_torque": eq, "median_field_T": B_med,
               "thetas_rad": list(s["thetas"]), "m_scales": list(s["m_scales"]),
               "n_axes": s["n_axes"], "t_max_s": s["t_max"],
               "T_orbit_s": T_ORBIT,
               "diagnostic": diag, "families": fams, "raw": res}
    with open(f"{OUT}/B_planner_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}/B_planner_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

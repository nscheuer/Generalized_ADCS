"""P2.8 -- Planner robustness to plan-vs-sim model mismatch (Paper 2).

Single-trajectory demos of how the plan-and-track planner tolerates a mismatch
between the model it PLANS on and the plant it actually flies:

  * Inertia (J) mismatch: the controller plans + tracks on an estimated sat
    whose inertia is scaled (est J = s x true J); the plant uses the true J.
  * B-field mismatch: the planner plans on the nominal geomagnetic field, but
    the realized (plant + onboard-sensed) field is scaled by s (and, for one
    case, rotated). The magnetorquer torque tau = m x B is therefore off from
    what the plan assumed.

3+1, ECI-pointing acquisition, tf=1000 s (the trackable regime). One trajectory
per mismatch level; reports pointing-error vs time + final/settling vs level.

Emits:
  output_data/P2.8_mismatch_<ts>.json
  output_data/fig_mismatch.{png,pdf}   (J panel | B panel)
  P2.8_RESULTS.md (separate).
"""

import os
import sys
import json
import datetime as _dt

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.orbit import Orbit
import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
CONFIG, CONV, SEC2CENT = "3+1", 5.0, TimeConstants.sec2cent
GOAL_VEC = normalize(np.array([0.3, 0.6, 0.74]))
J_SCALES = [0.7, 0.85, 1.0, 1.15, 1.3]
B_SCALES = [0.7, 0.85, 1.0, 1.15, 1.3]
BROT_ANGLES = [0.0, 15.0, 30.0, 45.0, 60.0]       # field-rotation error [deg]

# kind -> (levels, legend title, nominal level)
KINDS = {
    "J":    (J_SCALES,    "est J / true J",       1.0),
    "B":    (B_SCALES,    "plant B / planned B",  1.0),
    "Brot": (BROT_ANGLES, "field rotation [deg]", 0.0),
}


def _bore(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        b = b.get("default", next(iter(b.values()))) if isinstance(b, dict) else b
    return normalize(np.asarray(b, float).reshape(3))


def run_episode(kind, level, tf=1000.0, dt=1.0):
    """One trajectory under a J-scale or B-scale (or B-rotation) mismatch.
    Returns (t, pointing_error_deg)."""
    true_sat = P.make_sat(CONFIG, estimated=False)
    bu = _bore(true_sat)

    # Controller plans/tracks on est_sat; for J mismatch est J = level * true J.
    est_sat = P.make_sat(CONFIG, estimated=False)
    if kind == "J":
        est_sat.update_J(J_0=level * np.asarray(true_sat.J_0, float))
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=est_sat, planner_settings=P.make_planner_settings(est_sat))

    os0 = P.default_os0()
    orb = Orbit(os0=os0, end_time=os0.J2000 + (tf + 5) * SEC2CENT, dt=dt,
                use_J2=True, fast=False, verbose=False)
    N = int(tf / dt)
    os_seq = [orb.get_os(J2000=os0.J2000 + k * dt * SEC2CENT) for k in range(N + 1)]

    goal = ADCS.goals.ECI_Goal(GOAL_VEC)
    gl = ADCS.GoalList(goal_timeline={0.0: goal}, time_units="seconds",
                       start_juliantime=os0.J2000)
    x = P.x0(1)

    # Plan on the nominal field / est inertia.
    traj = ctrl.calculate_trajectory(os0.J2000, tf, x, os0, gl)
    ctrl.set_active_trajectory(traj)

    # B-field mismatch: scale (and optionally rotate) the realized field.
    Brot = None
    if kind == "Brot":
        ang = np.radians(level)  # level interpreted as rotation angle [deg]
        Brot = rot_mat(np.concatenate([[np.cos(ang / 2)],
                                       np.sin(ang / 2) * normalize(np.array([1.0, 1.0, 0.5]))]))

    def perturbed_os(os_k):
        if kind == "B":
            o = os_k.copy(); o.B = level * np.asarray(os_k.B, float); return o
        if kind == "Brot":
            o = os_k.copy(); o.B = Brot @ np.asarray(os_k.B, float); return o
        return os_k

    t_hist, err_hist = [], []
    for k in range(N):
        os_k = perturbed_os(os_seq[k])
        ag = gl.get_active_goal(os_k.J2000, time_units="centuries")
        err_hist.append(_angle_deg(_boresight_eci(x[3:7], bu),
                                   ag.to_ref(os_k)[0][1:4]))
        t_hist.append(k * dt)
        u = ctrl.find_u(x_hat=x, sens=true_sat.sensor_readings(x=x, os=os_k),
                        est_sat=est_sat, os_hat=os_k, goal=ag)
        os_next = perturbed_os(os_seq[k + 1])
        out = solve_ivp(fun=true_sat.dynamics_for_solver, t_span=(0, dt), y0=x,
                        method="RK45", args=(u, os_k, os_next),
                        rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]; x[3:7] = normalize(x[3:7])
    return np.asarray(t_hist), np.asarray(err_hist)


def settle(t, err):
    above = err > CONV
    if not above.any():
        return float(t[0])
    if above[-1]:
        return float("nan")
    return float(t[int(np.flatnonzero(above)[-1]) + 1])


def main():
    ts = os.environ.get("P28_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    tf = float(os.environ.get("P28_TF", 1000.0))
    kinds = os.environ.get("P28_KINDS", "J,B,Brot").split(",")
    titles = {"J": "inertia (J) mismatch", "B": "B-field magnitude mismatch",
              "Brot": "B-field rotation mismatch"}
    print(f"[P2.8] config={CONFIG} tf={tf} kinds={kinds} ts={ts}")

    results = {}
    fig, axs = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.6),
                            sharey=True, squeeze=False)
    axs = axs[0]

    for ax, kind in zip(axs, kinds):
        levels, lbl, nominal = KINDS[kind]
        results[kind] = []
        for s in levels:
            t, err = run_episode(kind, s, tf=tf)
            rec = {"level": s, "final_deg": float(err[-1]),
                   "settle_s": settle(t, err),
                   "max_after_50pct_deg": float(np.max(err[len(err) // 2:]))}
            results[kind].append(rec)
            is_nom = abs(s - nominal) < 1e-9
            ax.plot(t, err, "-" if is_nom else "--", lw=2.2 if is_nom else 1.3,
                    label=f"{s:g}" + (" (nominal)" if is_nom else ""))
            print(f"  {kind} {s:g}: final {rec['final_deg']:.2f} deg  "
                  f"settle {rec['settle_s']}  max2nd {rec['max_after_50pct_deg']:.2f}")
        ax.axhline(CONV, c="k", ls=":", lw=0.7)
        ax.set_yscale("log"); ax.set_xlabel("time [s]"); ax.set_title(titles[kind])
        ax.legend(title=lbl, fontsize=8); ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("pointing error [deg]")
    fig.suptitle(f"P2.8: planner robustness to plan-vs-sim mismatch ({CONFIG})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_mismatch.png", dpi=150); fig.savefig(f"{OUT}/fig_mismatch.pdf")
    plt.close(fig)

    payload = {"task": "P2.8_mismatch", "timestamp": ts, "config": CONFIG,
               "tf": tf, "conv_threshold_deg": CONV,
               "goal_eci": GOAL_VEC.tolist(), "results": results}
    with open(f"{OUT}/P2.8_mismatch_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[P2.8] wrote {OUT}/P2.8_mismatch_{ts}.json + fig_mismatch")
    return payload


if __name__ == "__main__":
    main()

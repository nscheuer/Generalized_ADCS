"""Campaign B -- equal-peak-torque ablation. The highest-value single result in the campaign.

**Purpose.** Isolate rank-3 restoration from raw torque authority. Without this the agility
result reads as "a wheel with 300x the torque is faster", which nobody needs a paper to learn.

The 3MTQ+0RW bus is scaled up until its peak achievable torque equals the wheel's, then both
buses run identical rest-to-rest slews. If the slope difference survives at *equal torque*, it
is the rank restoration doing the work, not the authority.

Three corrections to the spec, all of which change the answer:

**1. "Equal peak torque" was under-defined by 1.4-1.6x.** Three magnetorquers under a box
constraint ``|m_i| <= M`` do not have peak torque ``M|B|``. The maximum over the box is attained
at a vertex ``m = M s``, ``s in {+-1}^3``, giving ``M|B| sqrt(3 - (s.B_hat)^2)`` -- between
``sqrt(2) M|B|`` (field along a body axis) and ``sqrt(8/3) M|B| ~ 1.63 M|B|`` (field along the
body diagonal). So the naive ``m_max = tau_w / B ~ 67`` A m^2 overstates the scaling by ~40%.
Here ``M`` is solved so the **orbit median** of the true box maximum equals ``tau_w``, and the
min/max over the orbit are reported so the residual ambiguity is visible rather than hidden.

The resulting dipole is physically absurd. That is intentional and must be labelled as such in
the paper -- this is an ablation, not a design proposal.

**2. Random slew axes cannot show the effect.** At equal peak torque the perpendicular-to-B
directions have *more* torque available than the wheel, so a random-axis sample is dominated by
fast cross-field slews and the fitted slope flattens toward 1/2 regardless of mechanism. Each
slew axis is therefore classified by ``|e_slew . B_hat|`` at slew start and the two families are
fitted **separately**. The along-field family is the headline; cross-field is the control.

**3. The spec's Theta range sits outside the regime the theory describes.** The exponent
(1/4 per the IV-B derivation; the spec's 1/3 is superseded) is a small-time statement about
the weight-3 growth structure; at 3 rad on a magnetic-only
bus the slew time approaches the field-rotation timescale and the power law must break. Theta
= 0.05 and 0.1 rad are added to buy small-Theta leverage, and curvature at the top end is
expected rather than surprising.

Predicted: slope ~1/2 for 3MTQ+1RW (double integrator, full authority) and ~1/4 for 3MTQ+0RW (IV-B derivation, superseding the spec's 1/3)
in the along-field-limited directions.

Disturbances off, truth state -- this is a clean kinematic test.

Run: ``python papers/IAC_1RW/generate_B_equal_torque.py``   (``B_SCALE=fast`` for a smoke run)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory import create_iac_6u_bus, IAC_6U

from papers.IAC_1RW._iac_sim import EPOCH, T_ORBIT, _get_orbit, rv_circular

OUT = os.path.join(os.path.dirname(__file__), "output_data")

#: Slew magnitudes [rad]. 0.05 and 0.1 are additions -- see the module docstring.
THETAS_RAD = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0)

#: Axes per Theta per bus. Classified after the fact by alignment with the field.
N_AXES = 12

#: A slew axis counts as along-field above this alignment, cross-field below the complement.
#: The band between them is dropped rather than forced into one family.
ALIGN_HI = 0.7
ALIGN_LO = 0.3

#: Rest-to-rest completion: within this angle AND essentially stopped.
DONE_ANGLE_RAD = np.deg2rad(2.0)
DONE_RATE_RPS = np.deg2rad(0.02)

#: Give up after this long. Magnetic-only along-field slews are slow by construction.
T_MAX_S = 3000.0

KP, KD, KC = 5e-5, 1e-3, 1e-3

SCALES = {"fast": {"n_axes": 3, "thetas": (0.1, 0.5, 2.0)},
          "paper": {"n_axes": N_AXES, "thetas": THETAS_RAD}}


def scale():
    return SCALES[os.environ.get("B_SCALE", "paper")]


# ---------------------------------------------------------------------------------------
# Equal-peak-torque scaling
# ---------------------------------------------------------------------------------------

_VERTICES = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)],
                     dtype=float)


def mtq_peak_torque(B: np.ndarray, m_max: float) -> float:
    """max_{|m_i| <= m_max} |m x B|, exactly.

    The maximum of a linear map over a box is attained at a vertex, so this is a max over the
    eight sign patterns -- **not** ``m_max |B|``, which is the single-axis answer and 1.4-1.6x
    too small.
    """
    return float(m_max * np.max(np.linalg.norm(np.cross(_VERTICES, B), axis=1)))


def field_samples(n: int = 180) -> np.ndarray:
    """Body-frame field over one orbit for an inertially-fixed attitude (identity)."""
    ephem = Ephemeris()
    out = np.zeros((n, 3))
    for k, t in enumerate(np.linspace(0.0, T_ORBIT, n, endpoint=False)):
        R, V = rv_circular(2.0 * np.pi * t / T_ORBIT, 97.0, 0.0)
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t * TimeConstants.sec2cent, R=R, V=V)
        out[k] = np.asarray(os_k.B, float)
    return out


def solve_equal_torque_m_max(B_samples: np.ndarray, tau_target: float) -> Dict[str, float]:
    """Find m_max whose ORBIT-MEDIAN peak torque equals tau_target."""
    def med(m):
        return float(np.median([mtq_peak_torque(B, m) for B in B_samples])) - tau_target

    m = brentq(med, 1e-3, 1e6, xtol=1e-6, rtol=1e-10, maxiter=200)
    peaks = np.array([mtq_peak_torque(B, m) for B in B_samples])
    return {"m_max": float(m),
            "median_peak_Nm": float(np.median(peaks)),
            "min_peak_Nm": float(peaks.min()),
            "max_peak_Nm": float(peaks.max()),
            "naive_m_max": float(tau_target / np.median(np.linalg.norm(B_samples, axis=1))),
            }


# ---------------------------------------------------------------------------------------
# One rest-to-rest slew
# ---------------------------------------------------------------------------------------

def slew_config(run_id: int, *, bus: str, theta: float, axis_seed: int,
                m_max: float) -> Dict[str, Any]:
    rng = np.random.default_rng(axis_seed)
    e = normalize(rng.standard_normal(3))
    q0 = normalize(rng.standard_normal(4))
    dq = np.concatenate([[np.cos(theta / 2.0)], e * np.sin(theta / 2.0)])
    q_goal = normalize(quat_mult(q0, dq))
    return {"run_id": run_id, "bus": bus, "theta": float(theta),
            "axis_seed": axis_seed, "m_max": float(m_max),
            "q0": q0, "q_goal": q_goal, "slew_axis_body": e,
            "raan_deg": float(rng.uniform(0, 360)), "phase_deg": float(rng.uniform(0, 360)),
            "inc_deg": 97.0, "n_rw": 1 if bus == "3+1" else 0,
            "tf": T_MAX_S, "dt": 1.0, "task": "full"}


def run_slew(config: Dict[str, Any]) -> Dict[str, Any]:
    slot = claim_worker_slot()
    try:
        n_rw = config["n_rw"]
        dt = config["dt"]
        steps = int(T_MAX_S / dt)

        kw = {"disturbances": (), "estimate_dipole": False}
        if n_rw == 0:
            kw["m_max"] = config["m_max"]
        sat = create_iac_6u_bus(n_rw=n_rw, **kw)

        orb = _get_orbit(config, dt, T_MAX_S)
        goal = Fixed_Attitude_Goal(np.asarray(config["q_goal"], float))
        ctrl = MTQ_w_RW_LP(est_sat=sat, p_gain=KP, d_gain=KD, c_gain=KC,
                           h_target=np.zeros(3))

        x = np.concatenate([np.zeros(3), config["q0"], np.zeros(n_rw)])
        qr = normalize(np.asarray(config["q_goal"], float))

        # Field alignment of the slew axis at slew start -- the classifier.
        os0 = orb.get_os(J2000=EPOCH)
        from ADCS.helpers.math_helpers import rot_mat
        B_body0 = rot_mat(config["q0"]).T @ np.asarray(os0.B, float)
        align = float(abs(np.asarray(config["slew_axis_body"], float)
                          @ (B_body0 / np.linalg.norm(B_body0))))

        t = 0.0
        t_done = np.nan
        for i in range(steps):
            if i % 200 == 0:
                update_worker_progress(slot, config["run_id"], i, steps)
            os_k = orb.get_os(J2000=EPOCH + t * TimeConstants.sec2cent)
            sens = sat.sensor_readings(x=x, os=os_k)
            u = np.asarray(ctrl.find_u(x_hat=x, sens=sens, est_sat=sat,
                                       os_hat=os_k, goal=goal), float)

            ang = 2.0 * np.arccos(np.clip(abs(float(x[3:7] @ qr)), 0.0, 1.0))
            if ang < DONE_ANGLE_RAD and np.linalg.norm(x[0:3]) < DONE_RATE_RPS:
                t_done = t
                break

            t += dt
            os_n = orb.get_os(J2000=EPOCH + t * TimeConstants.sec2cent)
            x = solve_ivp(sat.dynamics_for_solver, (0, dt), x, method="RK45",
                          args=(u, os_k, os_n), rtol=1e-6, atol=1e-6).y[:, -1]
            x[3:7] = normalize(x[3:7])

        return {"run_id": config["run_id"], "bus": config["bus"],
                "theta": config["theta"], "alignment": align,
                "t_done_s": float(t_done), "completed": bool(np.isfinite(t_done))}
    finally:
        release_worker_slot(slot)


# ---------------------------------------------------------------------------------------

def fit_slope(thetas: np.ndarray, times: np.ndarray) -> Dict[str, float]:
    """log t_f = a + b log Theta. b is the exponent the theory predicts."""
    ok = np.isfinite(times) & (times > 0) & (thetas > 0)
    if ok.sum() < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n": int(ok.sum())}
    x, y = np.log(thetas[ok]), np.log(times[ok])
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss = 1.0 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    return {"slope": float(b), "intercept": float(a), "r2": ss, "n": int(ok.sum())}


def main() -> int:
    s = scale()
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 92)
    print("Campaign B -- equal-peak-torque ablation")
    print("=" * 92)

    Bs = field_samples()
    eq = solve_equal_torque_m_max(Bs, IAC_6U.tau_w)
    print(f"\nEqual-peak-torque scaling for the 3MTQ+0RW bus:")
    print(f"  m_max = {eq['m_max']:.2f} A m^2  (naive tau_w/|B| would give "
          f"{eq['naive_m_max']:.2f}, {eq['naive_m_max']/eq['m_max']:.2f}x too large)")
    print(f"  peak torque over the orbit: median {eq['median_peak_Nm']*1e3:.3f}, "
          f"min {eq['min_peak_Nm']*1e3:.3f}, max {eq['max_peak_Nm']*1e3:.3f} mN m "
          f"(target {IAC_6U.tau_w*1e3:.3f})")
    print("  PHYSICALLY ABSURD BY CONSTRUCTION -- this is an ablation, not a design.")

    cfgs: List[Dict[str, Any]] = []
    rid = 0
    for bus in ("3+1", "3+0"):
        for th in s["thetas"]:
            for k in range(s["n_axes"]):
                cfgs.append(slew_config(rid, bus=bus, theta=th,
                                        axis_seed=100000 + 1000 * k + int(th * 100),
                                        m_max=eq["m_max"]))
                rid += 1

    print(f"\nrunning {len(cfgs)} slews "
          f"({len(s['thetas'])} thetas x {s['n_axes']} axes x 2 buses)...")
    runner = MonteCarloRunner(sim_func=run_slew,
                              config_generator=lambda i, _c=cfgs: _c[i],
                              num_runs=len(cfgs))
    res = [r for r in runner.run() if r is not None]

    # ---- classify and fit ------------------------------------------------------------
    out: Dict[str, Any] = {}
    print(f"\n{'bus':<6}{'family':<14}{'n':>5}{'slope':>9}{'r2':>8}   predicted")
    print("-" * 92)
    for bus in ("3+1", "3+0"):
        for fam, lo, hi in (("along-field", ALIGN_HI, 1.01), ("cross-field", -0.01, ALIGN_LO)):
            sel = [r for r in res if r["bus"] == bus and lo <= r["alignment"] < hi]
            th = np.array([r["theta"] for r in sel])
            tt = np.array([r["t_done_s"] for r in sel])
            f = fit_slope(th, tt)
            key = f"{bus}|{fam}"
            out[key] = {"fit": f, "n_total": len(sel),
                        "n_completed": int(np.isfinite(tt).sum()),
                        "points": [{"theta": float(a), "t": float(b), "align": float(c)}
                                   for a, b, c in zip(th, tt, [r["alignment"] for r in sel])]}
            pred = ("~1/2 (double integrator)" if bus == "3+1"
                    else ("~1/4 (IV-B derivation; supersedes the spec's 1/3)" if fam == "along-field" else "~1/2"))
            print(f"{bus:<6}{fam:<14}{f['n']:>5}{f['slope']:>9.3f}{f['r2']:>8.3f}   {pred}")

    payload = {"task": "B_equal_torque", "timestamp": ts,
               "equal_torque": eq, "thetas_rad": list(s["thetas"]),
               "n_axes": s["n_axes"], "align_hi": ALIGN_HI, "align_lo": ALIGN_LO,
               "done_angle_rad": DONE_ANGLE_RAD, "done_rate_rps": DONE_RATE_RPS,
               "t_max_s": T_MAX_S, "families": out,
               "raw": res}
    with open(f"{OUT}/B_equal_torque_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("=" * 92)
    print(f"\nwrote {OUT}/B_equal_torque_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

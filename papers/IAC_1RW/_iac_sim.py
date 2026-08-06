"""Shared simulation core for the IAC campaign (A, C, E).

One worker, one metric set, one place where the estimator and the field-model error are wired.
Campaigns parametrise it rather than copying it, so a change to the bus or the sensing chain
cannot silently apply to some cells and not others.

What is always on here, unlike the companion papers:

* **Estimation is in the loop** -- UAKF over the augmented state (attitude, rate, wheel
  momentum, gyro + magnetometer bias, and the residual dipole). The frontier should be limited
  by *actuation*, not by the attitude solution.
* **The estimator and the plant see different magnetic fields** (``_field_error``). Without
  this the residual-dipole estimate is limited only by magnetometer noise, and the Section IV
  cancellation result comes out optimistic.
* **Full disturbance set** with the cp-cg offset, so drag and SRP are not identically zero.

Campaigns B, C and D run on truth states by design; pass ``use_estimator=False``.

Both reporting horizons come from **one** trajectory: metrics are evaluated at 1000 s and at
one full orbit from the same run, which halves the cost and makes the two exactly paired.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal
from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.mc.monte_carlo_runner import (
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_factory import create_iac_6u_bus, IAC_6U
from ADCS.satellite_factory.sensors import IAC_SENSOR_SPEC

from papers.IAC_1RW._field_error import FieldErrorModel, wrap_os_for_gnc

ALT_KM = 400.0
A_KM = EarthConstants.R_e + ALT_KM
T_ORBIT = 2.0 * np.pi * np.sqrt(A_KM ** 3 / EarthConstants.mu_e)   # 5553.6 s
INC_DEG = 97.0
EPOCH = 0.22

#: Reporting horizons. The 1000 s figure exists so the paper is legible against the companion
#: papers; the one-orbit figure is the one the argument rests on.
HORIZONS_S = (1000.0, T_ORBIT)

#: Coarse-solution initialisation error handed to the fine filter (sun sensor + magnetometer
#: class). The initial covariance below is sized to match, so the filter is neither
#: overconfident (which diverges) nor uninformative.
INIT_ATT_ERR_DEG = 10.0
INIT_RATE_ERR_DPS = 0.05

#: Initial body rate, deg/s. Lowered from U(0.1, 1.0): this campaign starts from an
#: already-detumbled, operating spacecraft re-targeting between observations, not from a
#: dispenser release. The old spread also drove the star tracker through its 2 deg/s rate
#: limit during acquisition, which cost availability exactly when the filter needed it.
INIT_RATE_DPS_RANGE = (0.05, 0.3)

#: Baseline stored wheel momentum as a fraction of h_max. A real momentum-biased bus does not
#: fly its wheel at zero -- it sits at a working point so the wheel can accept momentum in
#: either direction and so the stored momentum provides gyroscopic stiffness about the two
#: axes the single wheel cannot actuate. Campaign C varies this deliberately; every other
#: campaign now starts here rather than at rest.
BASELINE_H_FRAC = 1.0 / 3.0

_CACHED_ORBIT = None
_CACHED_KEY = None


# ----------------------------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------------------------

def rv_circular(u_rad: float, inc_deg: float, raan_deg: float):
    i, Om = np.deg2rad(inc_deg), np.deg2rad(raan_deg)
    v = np.sqrt(EarthConstants.mu_e / A_KM)
    cu, su, cO, sO, ci, si = (np.cos(u_rad), np.sin(u_rad), np.cos(Om),
                              np.sin(Om), np.cos(i), np.sin(i))
    r_hat = np.array([cO * cu - sO * ci * su, sO * cu + cO * ci * su, si * su])
    v_hat = np.array([-cO * su - sO * ci * cu, -sO * su + cO * ci * cu, si * cu])
    return A_KM * r_hat, v * v_hat


def make_config(run_id: int, *, n_rw: int, task: str, tf: float = T_ORBIT,
                dt: float = 1.0, seed: Optional[int] = None,
                inc_deg: float = INC_DEG, **extra) -> Dict[str, Any]:
    """One trial's scenario.

    Config-independent draws come first so that cells sharing a seed see the *same* scenario
    (attitude, rate, goal, orbit); anything whose size depends on the actuator complement is
    drawn last so it cannot shift the shared stream.
    """
    s = run_id if seed is None else seed
    rng = np.random.default_rng(seed=s)

    q0 = normalize(rng.standard_normal(4))
    rate_dps = rng.uniform(*INIT_RATE_DPS_RANGE)
    w0 = normalize(rng.standard_normal(3)) * rate_dps * np.pi / 180.0
    goal_vec = normalize(rng.standard_normal(3))
    q_goal = normalize(rng.standard_normal(4))
    raan = rng.uniform(0.0, 360.0)
    phase = rng.uniform(0.0, 360.0)
    field_seed = int(rng.integers(0, 2 ** 31 - 1))

    # Baseline momentum bias along each wheel axis (Campaign C overrides this).
    h0 = np.full(n_rw, BASELINE_H_FRAC * IAC_6U.h_max)

    cfg = {
        "run_id": run_id, "seed": s, "n_rw": n_rw, "task": task,
        "tf": float(tf), "dt": float(dt),
        "q0": q0, "w0": w0, "h0": h0,
        "goal_vec": goal_vec, "goal_quat": q_goal,
        "raan_deg": raan, "phase_deg": phase, "inc_deg": inc_deg,
        "field_seed": field_seed,
    }
    cfg.update(extra)
    return cfg


def build_goal(config: Dict[str, Any]):
    """``task='full'`` -> 3-axis attitude; ``task='reduced'`` -> boresight-to-inertial."""
    if config["task"] == "full":
        return Fixed_Attitude_Goal(np.asarray(config["goal_quat"], float))
    return ECI_Goal(np.asarray(config["goal_vec"], float))


def _get_orbit(config, dt, tf):
    global _CACHED_ORBIT, _CACHED_KEY
    key = (config["raan_deg"], config["phase_deg"], config["inc_deg"], dt, tf)
    if _CACHED_ORBIT is None or _CACHED_KEY != key:
        R, V = rv_circular(np.deg2rad(config["phase_deg"]), config["inc_deg"],
                           config["raan_deg"])
        os0 = Orbital_State(ephem=Ephemeris(), J2000=EPOCH - dt * TimeConstants.sec2cent,
                            R=R, V=V)
        _CACHED_ORBIT = Orbit(os0=os0, end_time=EPOCH + (tf + 2 * dt) * TimeConstants.sec2cent,
                              dt=dt, zonal_J=2, fast=False, verbose=False)
        _CACHED_KEY = key
    return _CACHED_ORBIT


def make_estimator(est_sat, config, dt):
    """UAKF over the augmented state.

    There is no MEKF in this codebase -- the estimators are UAKF and SRUAKF. The UAKF is used
    here and cited as a UKF; SRUAKF has recorded drifts relative to it.
    """
    n_rw = config["n_rw"]
    # x_hat is the FULL augmented state (base + biases + disturbance params); P/Q are one
    # shorter because the quaternion carries 3 DOF, not 4.
    n_bias = est_sat.act_bias_len + est_sat.att_sens_bias_len
    n_dist = est_sat.dist_param_len

    # Initialise from a COARSE attitude solution, not from the identity quaternion.
    #
    # A real spacecraft hands its fine filter a coarse estimate from sun sensor plus
    # magnetometer (a few degrees to ~10). Starting at identity while the truth is a uniform
    # draw on SO(3) means a ~90-180 degree initial error against a covariance that claims a
    # few degrees -- the filter is grossly overconfident, the UKF cannot recover, and every
    # cell diverges for a reason that has nothing to do with the actuators the paper is
    # about. The initial error below is drawn from the same seed as the scenario, so cells
    # sharing a seed also share their initial estimate.
    rng = np.random.default_rng(20_000_000 + int(config["seed"]))
    ang = np.deg2rad(INIT_ATT_ERR_DEG) * rng.standard_normal()
    axis = normalize(rng.standard_normal(3))
    dq = np.concatenate([[np.cos(ang / 2.0)], axis * np.sin(ang / 2.0)])
    q_hat0 = normalize(quat_mult(np.asarray(config["q0"], float), dq))
    w_hat0 = (np.asarray(config["w0"], float)
              + np.deg2rad(INIT_RATE_ERR_DPS) * rng.standard_normal(3))

    x_hat0 = np.concatenate([w_hat0, q_hat0,
                             np.asarray(config["h0"], float),
                             np.zeros(n_bias + n_dist)])
    reduced = len(x_hat0) - 1

    # Blocks: [rate(3) | attitude(3) | wheel(n_rw) | estimated biases | disturbance params].
    # Built from the satellite's own block sizes rather than hardcoded, so the campaign can
    # turn the dipole or a sensor off without silently mis-sizing the filter.
    #
    # Keep the spread of magnitudes across blocks narrow. A covariance spanning twelve orders
    # of magnitude is singular in float64 after a few hundred UKF updates regardless of the
    # physics -- which is why magnetometer bias is not carried (see create_iac_6u_bus).
    p_diag = np.concatenate([
        [np.deg2rad(INIT_RATE_ERR_DPS) ** 2] * 3,      # rate  [rad/s]^2
        [np.deg2rad(INIT_ATT_ERR_DEG) ** 2] * 3,       # attitude [rad]^2
        [max((0.01 * IAC_6U.h_max) ** 2, 1e-12)] * n_rw,   # wheel momentum, at tach sigma
        [(1e-4) ** 2] * n_bias,                        # gyro bias [rad/s]^2
        [(0.5 * IAC_6U.m_res) ** 2] * n_dist,          # dipole: 50% prior, deliberately loose
    ])
    q_diag = np.concatenate([
        [1e-12] * 3,
        [1e-12] * 3,
        [1e-14] * n_rw,
        # Gyro-bias process noise must MATCH the truth random walk, not be guessed.
        # create_iac_gyro drifts the bias by BI/sqrt(T_orbit/dt) per step so it reaches the
        # quoted bias instability over one orbit. A filter given a tighter Q than that is too
        # confident to track its own sensor: at 1e-16 it allowed 0.154 deg/hr against a truth
        # of 5 deg/hr -- 1058x too small in variance -- so the bias ran away and dragged the
        # attitude estimate to multi-degree error no matter how good the star tracker was.
        [(IAC_SENSOR_SPEC["gyro_bias_instab_rad_per_s"]
          / np.sqrt(T_ORBIT / dt)) ** 2] * n_bias,
        # Dipole process noise must not be starved: an over-tight Q collapses P_dist and the
        # filter simply stops learning the dipole (recorded failure mode from the P2.6 leak
        # work, where the measurement/dipole cross-covariance fell to ~1e-10 and the estimate
        # froze). Sized so the dipole can move by ~1% of its magnitude over an orbit.
        [(1e-3 * IAC_6U.m_res) ** 2] * n_dist,
    ])
    assert p_diag.size == reduced, (p_diag.size, reduced)

    return UAKF(est_sat=est_sat, J2000=EPOCH, x_hat=x_hat0,
                P_hat=np.diagflat(p_diag), Q_hat=np.diagflat(q_diag),
                dt=dt, cross_term=True, quat_as_vec=False)


# ----------------------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------------------

def simulate(config: Dict[str, Any],
             make_controller: Callable[[Any, Dict[str, Any]], Any],
             *,
             use_estimator: bool = True,
             disturbances=("gg", "drag", "srp", "dipole"),
             field_error_deg: float = 4.0,
             field_error_frac: float = 0.04,
             bus_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one trial and return histories plus per-step diagnostics."""
    slot_id = claim_worker_slot()
    run_id = config["run_id"]
    try:
        dt, tf = config["dt"], config["tf"]
        steps = int(round(tf / dt))
        n_rw = config["n_rw"]
        bus_kwargs = dict(bus_kwargs or {})

        sat = create_iac_6u_bus(n_rw=n_rw, disturbances=disturbances,
                                estimate_dipole=("dipole" in disturbances),
                                **bus_kwargs)
        if use_estimator:
            est_sat = create_iac_6u_bus(n_rw=n_rw, disturbances=disturbances,
                                        estimate_dipole=("dipole" in disturbances),
                                        estimated=True, **bus_kwargs)
            # The filter starts with **no** knowledge of the residual dipole. Seeding it with
            # the truth would make the Section IV cancellation result circular.
            for d in est_sat.disturbances:
                if getattr(d, "estimate_dist", False):
                    d.main_param = np.zeros(3)
            estimator = make_estimator(est_sat, config, dt)
        else:
            est_sat, estimator = sat, None

        field_err = FieldErrorModel(direction_deg=field_error_deg,
                                    magnitude_frac=field_error_frac,
                                    rng=np.random.default_rng(config["field_seed"]))

        orb = _get_orbit(config, dt, tf)
        goal = build_goal(config)
        controller = make_controller(est_sat, config)

        x = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(sat.rw_actuators):
            rw.h = config["h0"][i]

        t_hist = np.zeros(steps)
        state_hist = np.zeros((steps, len(x)))
        u_hist = np.zeros((steps, len(sat.actuators)))
        est_hist = np.full((steps, len(x)), np.nan)
        dip_hist = np.full((steps, 3), np.nan)
        avail_hist = np.zeros(steps, dtype=bool)
        # Eclipse per step. At 400 km SSO the spacecraft is in shadow for roughly a third of
        # the orbit; without a tracker fix that leaves magnetometer + gyro propagation, so a
        # bad-trial/eclipse correlation is a real sensor-suite finding rather than a bug and
        # has to be checkable per trial.
        eclipse_hist = np.zeros(steps, dtype=bool)

        trackers = [s for s in sat.sensors if hasattr(s, "available")]
        n_mtq = len(sat.mtq_actuators)
        m_max = sat.mtq_actuators[0].u_max if n_mtq else 1.0

        t = 0.0
        sec2cent = TimeConstants.sec2cent
        for i in range(steps):
            if i % 100 == 0:
                update_worker_progress(slot_id, run_id, i, steps)

            J2000 = EPOCH + t * sec2cent
            os_k = orb.get_os(J2000=J2000)
            sens = sat.sensor_readings(x=x, os=os_k)
            # ANY tracker with a fix counts -- the pair exists precisely so one can cover
            # while the other is blinded.
            avail_hist[i] = any(bool(tr.available) for tr in trackers) if trackers else True
            R_e = EarthConstants.R_e
            Rk = np.asarray(os_k.R, float); Sk = np.asarray(getattr(os_k, "S", None), float)
            if Sk is not None and Sk.size == 3 and np.linalg.norm(Sk) > 0:
                s_hat = Sk / np.linalg.norm(Sk)
                proj = float(Rk @ s_hat)
                eclipse_hist[i] = (proj < 0.0 and
                                   float(np.linalg.norm(Rk - proj * s_hat)) < R_e)

            # The GNC chain believes a slightly wrong field; the plant integrates the true one.
            os_gnc = wrap_os_for_gnc(os_k, field_err)

            if estimator is not None:
                x_hat = np.asarray(estimator.update(u=u_hist[i - 1] if i else np.zeros(len(sat.actuators)),
                                                    sensors=sens, os=os_gnc), float)
                est_hist[i, :] = x_hat[:len(x)]
                for d in est_sat.disturbances:
                    if getattr(d, "estimate_dist", False):
                        dip_hist[i, :] = np.ravel(d.main_param)[:3]
            else:
                x_hat = x

            u = controller.find_u(x_hat=x_hat, sens=sens, est_sat=est_sat,
                                  os_hat=os_gnc, goal=goal)
            u = np.asarray(u, float)

            t_hist[i] = t
            state_hist[i] = x
            u_hist[i] = u

            t += dt
            os_next = orb.get_os(J2000=EPOCH + t * sec2cent)
            sol = solve_ivp(fun=sat.dynamics_for_solver, t_span=(0, dt), y0=x,
                            method="RK45", args=(u, os_k, os_next), rtol=1e-6, atol=1e-6)
            x = sol.y[:, -1]
            x[3:7] = normalize(x[3:7])

        update_worker_progress(slot_id, run_id, steps, steps)
        return {
            "run_id": run_id, "config": config,
            "time": t_hist, "state": state_hist, "u": u_hist,
            "est": est_hist, "dipole_est": dip_hist, "tracker_available": avail_hist,
            "eclipse": eclipse_hist,
            "n_mtq": n_mtq, "m_max": m_max,
        }
    finally:
        release_worker_slot(slot_id)


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------

def error_series(run: Dict[str, Any]) -> np.ndarray:
    """Pointing error [deg]: 3-axis attitude error for 'full', boresight angle for 'reduced'."""
    cfg = run["config"]
    q = run["state"][:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if cfg["task"] == "full":
        qr = normalize(np.asarray(cfg["goal_quat"], float))
        return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(q @ qr), 0.0, 1.0)))

    # Reduced attitude: angle between the body boresight in ECI and the target direction.
    b = np.asarray(IAC_6U.boresight, float)
    tgt = normalize(np.asarray(cfg["goal_vec"], float))
    w, xq, yq, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Rotate the body boresight into ECI:  v = q (x) b (x) q*
    t = 2.0 * np.cross(np.column_stack([xq, yq, zq]), b)
    b_eci = b + w[:, None] * t + np.cross(np.column_stack([xq, yq, zq]), t)
    cosang = np.clip(b_eci @ tgt, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def acquire_time(t: np.ndarray, err: np.ndarray, thresh_deg: float) -> float:
    """First time the error goes below ``thresh`` **and stays there**. NaN if never."""
    below = err < thresh_deg
    if not below.any():
        return float("nan")
    # Walk back from the end to find the start of the final contiguous in-spec run.
    last_bad = np.where(~below)[0]
    idx = 0 if last_bad.size == 0 else last_bad[-1] + 1
    return float(t[idx]) if idx < t.size else float("nan")


def cell_metrics(runs: List[Dict[str, Any]], horizon_s: float,
                 hold_frac: float = 0.25) -> Dict[str, Any]:
    """Metrics for one cell at one horizon.

    ``hold_frac`` is the trailing fraction of the horizon treated as the held interval. The
    sustained statistic is what Campaign E's accuracy axis uses -- final-sample error is an
    artifact of where a replanning window boundary happens to land.
    """
    finals, held_p95, acq5, acq1 = [], [], [], []
    h_peak, h_final, duty, avail = [], [], [], []
    dip_frac, dip_sec_frac = [], []
    est_med, est_p95, ecl = [], [], []

    for r in runs:
        if r is None:
            continue
        t = r["time"]
        k = int(np.searchsorted(t, horizon_s, side="right"))
        if k < 2:
            continue
        err = error_series(r)[:k]
        tt = t[:k]

        finals.append(float(err[-1]))
        h0 = int((1.0 - hold_frac) * k)
        held_p95.append(float(np.percentile(err[h0:], 95)))
        acq5.append(acquire_time(tt, err, 5.0))
        acq1.append(acquire_time(tt, err, 1.0))

        n_rw = r["config"]["n_rw"]
        if n_rw:
            h = r["state"][:k, 7:7 + n_rw]
            hm = np.abs(h).max(axis=1) / IAC_6U.h_max
            h_peak.append(float(hm.max()))
            h_final.append(float(hm[-1]))

        n_mtq = r["n_mtq"]
        if n_mtq:
            m = np.linalg.norm(r["u"][:k, :n_mtq], axis=1) / (r["m_max"] * np.sqrt(n_mtq))
            duty.append(float(np.mean(m)))
        avail.append(float(np.mean(r["tracker_available"][:k])))
        e_hist = r.get("eclipse")
        ecl.append(float(np.mean(e_hist[:k])) if e_hist is not None else float("nan"))

        # ESTIMATED-attitude error, over the held interval, reported for EVERY cell.
        #
        # A pointing threshold below the knowledge floor measures the filter, not the
        # actuators: the controller cannot point better than it can be told where it is. If
        # this is comparable to the convergence threshold, the cell is not evidence about
        # the architecture and either the threshold rises or the sensor model needs work.
        est = r.get("est")
        if est is not None and est.shape[0] >= k:
            q_t = r["state"][h0:k, 3:7]
            q_e = est[h0:k, 3:7]
            good = np.isfinite(q_e).all(axis=1)
            if good.any():
                dots = np.abs(np.sum(q_t[good] * q_e[good], axis=1))
                ang = np.rad2deg(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
                est_med.append(float(np.median(ang)))
                est_p95.append(float(np.percentile(ang, 95)))

        # Residual-dipole cancellation quality, as a FRACTION of the original dipole.
        #
        # This is what Section IV-A's momentum argument turns on. If the estimate leaves,
        # say, a quarter of the dipole standing, the leftover torque sits at the same order
        # as drag -- and the claim that the momentum boundary is a drag story then rests
        # entirely on the residual being *cyclic* while drag is secular, which has to be
        # written down rather than assumed. So report both the magnitude and the split.
        dip = r.get("dipole_est")
        if dip is not None and dip.shape[0] >= k and np.isfinite(dip[:k]).any():
            m_true = np.asarray(IAC_6U.m_res_dir, float)
            m_true = IAC_6U.m_res * m_true / np.linalg.norm(m_true)
            # Use the held interval: the early transient is the filter converging, not the
            # steady cancellation quality the paper quotes.
            seg = dip[int((1.0 - hold_frac) * k):k]
            seg = seg[np.isfinite(seg).all(axis=1)]
            if seg.size:
                err = seg - m_true                       # per-step residual dipole vector
                dip_frac.append(float(np.linalg.norm(err, axis=1).mean()
                                      / np.linalg.norm(m_true)))
                # Secular part = the orbit-mean of the residual (what a body-fixed wheel
                # actually integrates); the rest is cyclic and averages out.
                sec = np.linalg.norm(err.mean(axis=0))
                rms = float(np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean()))
                dip_sec_frac.append(float(sec / rms) if rms > 0 else float("nan"))

    finals = np.asarray(finals)
    if finals.size == 0:
        return {"n": 0}

    def _nanmed(v):
        v = np.asarray(v, float)
        return float(np.nanmedian(v)) if v.size and not np.all(np.isnan(v)) else float("nan")

    return {
        "n": int(finals.size),
        "horizon_s": float(horizon_s),
        "conv_pct_5deg": float(100.0 * np.mean(finals < 5.0)),
        "conv_pct_1deg": float(100.0 * np.mean(finals < 1.0)),
        "mean_final_deg": float(np.mean(finals)),
        "median_final_deg": float(np.median(finals)),
        "p95_final_deg": float(np.percentile(finals, 95)),
        "median_held_p95_deg": _nanmed(held_p95),
        "median_acquire_5deg_s": _nanmed(acq5),
        "median_acquire_1deg_s": _nanmed(acq1),
        "acquired_5deg_frac": float(np.mean(~np.isnan(np.asarray(acq5, float)))),
        "median_peak_h_frac": _nanmed(h_peak) if h_peak else None,
        "median_final_h_frac": _nanmed(h_final) if h_final else None,
        "mean_mtq_duty": _nanmed(duty) if duty else None,
        "mean_tracker_available": float(np.mean(avail)) if avail else None,
        # Knowledge floor. Compare against the convergence thresholds before trusting them.
        "median_est_att_err_deg": _nanmed(est_med) if est_med else None,
        "p95_est_att_err_deg": _nanmed(est_p95) if est_p95 else None,
        # Fraction of the original residual dipole still standing after cancellation, and
        # how much of that leftover is secular (the part a body-fixed wheel integrates)
        # rather than cyclic.
        "median_dipole_residual_frac": _nanmed(dip_frac) if dip_frac else None,
        "median_dipole_residual_secular_frac": _nanmed(dip_sec_frac) if dip_sec_frac else None,
        "finals_deg": finals.tolist(),
        # Per-trial, so error-vs-availability and error-vs-eclipse correlations can be done
        # from disk without re-running anything.
        "per_trial_tracker_avail": avail,
        "per_trial_eclipse_frac": ecl,
        "per_trial_est_att_err_deg": est_med,
    }

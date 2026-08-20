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

Planner controllers (``Plan_and_Track_LQR``) are driven with the proven windowed-replanning
recipe rather than bare ``find_u``: plan ``window + overlap`` and execute only the window
(executing to a plan's endpoint produces joint spikes as the TVLQR gains shrink), wall-clock
timeout per plan, and a reactive PD fallback because ``trajOpt`` raises on non-convergence.
Plans are computed from the ESTIMATED state on the GNC-side field -- the planner is flight
software and sees what the filter sees, not the truth.

Both reporting horizons come from **one** trajectory: metrics are evaluated at 1000 s and at
one full orbit from the same run, which halves the cost and makes the two exactly paired.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal, Nadir_Goal
from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_helpers import (normalize, quat_mult, quat_inv,
                                       rot_mat as quat_to_rot)
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

#: Windowed-replanning parameters (validated in the planner paper: 500 s + 500 s overlap,
#: execute the first 500 only).
PLAN_WINDOW_S = 500.0
PLAN_OVERLAP_S = 500.0
# Hard wall budget per solve, enforced at a PROCESS boundary (see _plan_in_child).
# 300 s is >5x a typical money-cell solve (~50 s) and 170x under the observed pathology
# (14.5 h CPU, 2026-08-20): a solve needing more than 5x typical is operationally
# non-convergent and the designed answer is the fallback controller, counted per trial.
# (The old value, 90 s, was calibrated under SIGALRM -- which never actually fired inside
# the C++ solve, so it was never a real budget and never clipped anything.)
PLAN_TIMEOUT_S = 300.0


class _PlanTimeout(Exception):
    """Solve did not produce a usable trajectory (non-convergence / solver failure)."""


class _PlanBudgetKill(_PlanTimeout):
    """Solve was KILLED at the wall budget -- the solver hung, which is a different
    event from non-convergence: the problem might well have been solvable with more
    time. The two must never share a column (they have opposite implications for the
    planner measurement), so the call site counts them separately."""


def enforce_wheel_envelope(u, h, h_max, n_mtq, n_rw, dt=1.0):
    """The wheel driver: a wheel at its momentum limit cannot accelerate further.

    One-sided torque box at the limit -- at h >= +h_max only despin (u <= 0) is physically
    available, and symmetrically at -h_max. This is hardware truth, not a numerical clamp:
    the integrator previously ran a 15 mN m s wheel to 184% of its limit, and everything a
    trajectory does after that point is dynamics no hardware can produce (h-dot = u_rw -
    (wdot . a) J_rw, so positive u raises h). Applied in the harness with TRUE h and in the
    controller with ESTIMATED h, so the flight software cannot wind up against the driver.
    """
    u = np.asarray(u, float)
    for i in range(n_rw):
        hi = float(h[i])
        # Step-aware bound, not a boundary check: one dt at full torque moves h by
        # tau_w*dt = 13% of h_max, so gating only on the step-START h overshoots the limit
        # mid-step (caught at 1.012 h_max by the negative test). Bounding u to the torque
        # that just reaches the limit within the step is torque-then-coast physics, and it
        # collapses to the one-sided box exactly at the limit.
        # SIGN, determined empirically (3-step probe, u=+tau_w -> h=-6e-3): hdot = -u.
        # The command is torque ON THE BODY; the wheel stores the reaction. The kernel's
        # docstring u_rw is the wheel-internal torque, which misled the first version of
        # this bound into ACTIVELY PUMPING the wheel at the limit (h ran to 83 h_max under
        # the inverted clamp -- caught by the negative test, again).
        # Delta-h = -u*dt  =>  u in [(hi - h_max)/dt, (hi + h_max)/dt].
        u[n_mtq + i] = float(np.clip(u[n_mtq + i],
                                     (hi - h_max) / dt, (hi + h_max) / dt))
    return u


def _plan_in_child(seconds, fn, *a, **kw):
    """Run a solve in a forked child with a kill-on-overrun wall budget.

    SIGALRM cannot do this job: Python signals fire only between bytecodes, so a
    handler never runs while the C++ solver grinds -- the 90 s "timeout" this replaces
    was inert exactly when it was needed (observed 2026-08-20: 2 of 100 money-cell
    solves at 14.5 h CPU each, every alarm armed, none fired, pool wedged forever).
    A process boundary is the only budget the solver cannot ignore: the child solves
    and ships the picklable result through a pipe; on overrun the parent SIGKILLs it
    and raises _PlanTimeout, which the call site already converts into a fallback
    window -- the designed non-convergence path, counted in n_fallbacks.

    os.fork (not multiprocessing.Process) because MC workers are daemonic and may not
    have multiprocessing children; the OS does not care. Child exits via os._exit so
    it never runs the parent's atexit/finalizers.
    """
    import os as _os
    import select as _select
    import signal as _signal
    import struct as _struct
    import time as _time

    r, w = _os.pipe()
    pid = _os.fork()
    if pid == 0:
        _os.close(r)
        code = 0
        try:
            try:
                payload = pickle.dumps((True, fn(*a, **kw)),
                                       protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException as e:  # noqa: BLE001 -- child must never propagate
                payload = pickle.dumps((False, f"{type(e).__name__}: {e}"[:2000]))
            _os.write(w, _struct.pack("!Q", len(payload)))
            mv = memoryview(payload)
            while len(mv):
                mv = mv[_os.write(w, mv[:1 << 16]):]
        except BaseException:
            code = 1
        _os._exit(code)

    _os.close(w)
    deadline = _time.monotonic() + seconds
    chunks: list = []
    try:
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0.0:
                raise _PlanBudgetKill(f"solve exceeded {seconds:.0f} s wall budget")
            ready, _, _ = _select.select([r], [], [], min(remaining, 1.0))
            if not ready:
                continue
            buf = _os.read(r, 1 << 16)
            if not buf:
                break
            chunks.append(buf)
    finally:
        _os.close(r)
        if _os.waitpid(pid, _os.WNOHANG)[0] == 0:
            _os.kill(pid, _signal.SIGKILL)
            _os.waitpid(pid, 0)

    blob = b"".join(chunks)
    if len(blob) < 8:
        raise _PlanTimeout("solve child died without a result")
    (nbytes,) = _struct.unpack("!Q", blob[:8])
    if len(blob) - 8 != nbytes:
        raise _PlanTimeout("solve child sent a truncated result")
    ok, val = pickle.loads(blob[8:])
    if not ok:
        raise _PlanTimeout(f"solve failed in child: {val}")
    return val

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

#: Star-tracker mounting for the nadir-staring profile: anti-parallel to the payload
#: boresight, so it stares at zenith. Fixed at build time -- the controller never knows the
#: tracker exists, which is what keeps control and estimation separable.
ST_AXES_NADIR = [np.array([0.0, 0.0, -1.0])]

#: Baseline stored wheel momentum as a fraction of h_max. A real momentum-biased bus does not
#: fly its wheel at zero -- but the working point must sit BELOW the transverse ceiling
#: h <= tau_perp / omega_slew (~6 mN m s at 0.3 deg/s on the settled bus). The original 1/3
#: (5 mN m s) predates the ceiling derivation and sits essentially AT it, over it during
#: acquisition transients; every verified-good configuration used 5%. Campaign C sweeps the
#: fraction deliberately; everything else starts here.
BASELINE_H_FRAC = 0.05

#: The settled bus, asserted rather than assumed. Defaults have silently drifted twice
#: (the big wheel surviving in the factory; the pre-fix sensor grades), and each time the
#: cost was a campaign's worth of numbers on rejected hardware. Every campaign entry point
#: calls this before launching anything.
SETTLED_BUS = {"tau_w": 2.0e-3, "h_max": 15.0e-3, "m_max": 0.6, "m_res": 0.05,
               "com_offset_m": 0.02, "n_trackers": 2}


def assert_settled_bus() -> None:
    """Fail fast if the factory defaults are not the settled configuration."""
    errs = []
    for attr in ("tau_w", "h_max", "m_max", "m_res", "com_offset_m"):
        got, want = getattr(IAC_6U, attr), SETTLED_BUS[attr]
        if not np.isclose(got, want, rtol=1e-9):
            errs.append(f"{attr}: factory {got!r} != settled {want!r}")
    sat = create_iac_6u_bus(n_rw=1)
    n_st = sum(1 for x in sat.sensors
               if type(x).__name__ == "StarTrackerQuaternion")
    if n_st != SETTLED_BUS["n_trackers"]:
        errs.append(f"n_trackers: factory {n_st} != settled {SETTLED_BUS['n_trackers']}")
    if errs:
        raise AssertionError(
            "Factory defaults have drifted from the settled bus -- refusing to launch a "
            "campaign on unreviewed hardware:\n  " + "\n  ".join(errs))


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
    """Task -> goal.

    ``"nadir"``    Earth-staring, the mission the paper actually motivates. Also the profile
                   that makes a single star tracker work: mounted anti-parallel to the payload
                   it points at zenith permanently, 180 deg from nadir, so the 95.2 deg Earth
                   keep-out can never fire. Availability 0.84-1.00 by RAAN against 0.456 for an
                   inertial stare -- from mounting alone, with no coupling between control and
                   estimation.
    ``"reduced"``  boresight to a fixed inertial direction.
    ``"full"``     3-axis attitude.
    """
    task = config["task"]
    if task == "full":
        return Fixed_Attitude_Goal(np.asarray(config["goal_quat"], float))
    if task == "nadir":
        return Nadir_Goal()
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


def _dist_prior(est_sat):
    """Initial variance per estimable disturbance parameter, on its own physical scale."""
    from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
    out = []
    for j in est_sat.dist_param_inds:
        d = est_sat.disturbances[j]
        n = int(np.size(d.main_param))
        if isinstance(d, Dipole_Disturbance):
            out.append([(0.5 * IAC_6U.m_res) ** 2] * n)   # 50% of the dipole, loose
        else:
            out.append([(1e-6) ** 2] * n)                 # lumped torque, ~1 uN m scale
    return [np.concatenate(out)] if out else [np.zeros(0)]


def _dist_process(est_sat):
    """Process noise per estimable disturbance parameter.

    Must not be starved: an over-tight Q collapses P_dist and the filter stops learning the
    disturbance entirely (the recorded P2.6 failure, where the measurement/disturbance
    cross-covariance fell to ~1e-10 and the estimate froze). Sized so each parameter can
    move by ~0.1% of its own scale per step.
    """
    from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
    out = []
    for j in est_sat.dist_param_inds:
        d = est_sat.disturbances[j]
        n = int(np.size(d.main_param))
        if isinstance(d, Dipole_Disturbance):
            out.append([(1e-3 * IAC_6U.m_res) ** 2] * n)
        else:
            out.append([(1e-3 * 1e-6) ** 2] * n)
    return [np.concatenate(out)] if out else [np.zeros(0)]


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
        # Disturbance params are HETEROGENEOUS: the dipole is in A m^2 (~0.05) and the
        # lumped torque in N m (~1e-6). One shared variance would be wrong by ~1e9, so each
        # estimable disturbance gets a prior on its own scale.
        *_dist_prior(est_sat),
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
        *_dist_process(est_sat),
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

        # Planner detection + reactive fallback. The fallback exists because trajOpt RAISES
        # on non-convergence; a trial must degrade to feedback, not die.
        is_planner = hasattr(controller, "calculate_trajectory")
        fallback = None
        n_plans = n_fallbacks = 0
        # Fallbacks are NOT one population. Budget-kill = the solver HUNG (problem
        # possibly solvable, wall budget enforced); solve-failure = non-convergence
        # (problem hard); track-fallback = find_u failed mid-window. Opposite
        # implications for whether a cell is "a planner measurement" -- separate columns.
        n_budget_kills = n_solve_failures = n_track_fallbacks = 0
        next_replan = 0.0
        active = controller
        if is_planner:
            from papers.IAC_1RW._feedforward import FeedforwardLP
            _h0 = np.asarray(config["h0"], float)
            fallback = FeedforwardLP(
                est_sat=est_sat, p_gain=2.9e-4,
                d_gain=2.0 * np.sqrt(2.9e-4 * 0.13 / 2.0), c_gain=1e-3,
                h_target=(_h0[0] * np.array([0.0, 0.0, 1.0])) if _h0.size
                else np.zeros(3), mode="dipole")

        # A stored momentum exceeding the wheel's limit is a config bug, not a scenario -- it
        # happened once (h0 computed against a stale factory h_max while bus_kwargs said
        # otherwise) and produced a 100-trial cell of 68-degree medians before anything
        # complained. Physical impossibility is rejected here, where the wheel is loaded.
        _h0_chk = np.asarray(config["h0"], float)
        if n_rw:
            _hmax_chk = float(np.ravel(sat.rw_actuators[0].h_max)[0])
            if _h0_chk.size and np.any(np.abs(_h0_chk) > _hmax_chk * (1.0 + 1e-9)):
                raise ValueError(
                    f"h0 {_h0_chk} exceeds the bus h_max {_hmax_chk} -- stored momentum "
                    f"computed against a different wheel than the one being flown")

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
        # Nadir direction per step. The pointing metric for a nadir-staring task needs the
        # target direction at each instant, and it is not recoverable from the state alone.
        nadir_hist = np.zeros((steps, 3))
        # --- geometry covariates, logged so the divergence correlation falls out of the
        # --- dataset instead of needing its own campaign afterwards.
        sigma_hist = np.zeros(steps)      # |a_hat . B_hat|: rank restoration needs sigma > 0
        alpha_hist = np.full(steps, np.nan)   # LP scale: how much of the request was deliverable
        hfrac_hist = np.zeros(steps)      # wheel momentum as a fraction of h_max
        Bmag_hist = np.zeros(steps)       # |B| -- gives transverse authority tau_perp
        omega_hist = np.zeros(steps)      # |omega| -- the damping-saturation covariate
        # Plan-vs-executed divergence [rad], planner cells only: the planner-side alpha.
        # Separates "the plan failed" (executed follows the plan, plan misses the goal) from
        # "the plan was fine and the tracker lost it" (executed departs the plan). Without it
        # a loose-TVLQR pathology in the diverged trials would masquerade as a planning limit
        # and muddy the frontier comparison.
        plan_dev_hist = np.full(steps, np.nan)
        # Decomposition of the deviation along/perpendicular to B (body frame). Along-field
        # concentration = the tracker fighting exactly the direction the magnetorquers cannot
        # serve (TVLQR underweighting the wheel's tracking authority); isotropic = globally
        # soft weights. Different retune in each case.
        plan_dev_alongB_hist = np.full(steps, np.nan)
        plan_dev_perpB_hist = np.full(steps, np.nan)

        h_max_eff = (float(np.ravel(sat.rw_actuators[0].h_max)[0])
                     if n_rw else IAC_6U.h_max)
        trackers = [s for s in sat.sensors if hasattr(s, "available")]
        n_mtq = len(sat.mtq_actuators)
        m_max = sat.mtq_actuators[0].u_max if n_mtq else 1.0

        B_body0 = np.zeros(3)
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
            Rk = np.asarray(os_k.R, float)
            nadir_hist[i] = -Rk / np.linalg.norm(Rk)
            B_b = np.ravel(os_k.get_state_vector(x=x)["b"])[:3]
            bn = float(np.linalg.norm(B_b))
            if i == 0:
                B_body0 = B_b.copy()
            Bmag_hist[i] = bn
            omega_hist[i] = float(np.linalg.norm(x[0:3]))
            if bn > 0 and n_rw:
                a_hat = np.ravel(sat.rw_actuators[0].axis)[:3]
                sigma_hist[i] = abs(float(a_hat @ B_b / bn))
            if n_rw:
                hfrac_hist[i] = float(np.abs(x[7:7 + n_rw]).max() / h_max_eff)
            Sk = np.asarray(getattr(os_k, "S", None), float)
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

            if is_planner and t >= next_replan:
                from ADCS.CONOPS.goallist import GoalList
                try:
                    traj = _plan_in_child(
                        PLAN_TIMEOUT_S, controller.calculate_trajectory,
                        os_gnc.J2000, PLAN_WINDOW_S + PLAN_OVERLAP_S,
                        np.asarray(x_hat, float).copy()[:len(x)], os_gnc,
                        GoalList({os_gnc.J2000: goal}))
                    controller.set_active_trajectory(traj)
                    active, n_plans = controller, n_plans + 1
                except _PlanBudgetKill as e:
                    active, n_fallbacks = fallback, n_fallbacks + 1
                    n_budget_kills += 1
                    print(f"[seed {config.get('seed', '?')}] PLAN BUDGET-KILL at "
                          f"t={t:.0f}s: {e}", flush=True)
                except Exception as e:
                    active, n_fallbacks = fallback, n_fallbacks + 1
                    n_solve_failures += 1
                    print(f"[seed {config.get('seed', '?')}] PLAN SOLVE-FAILURE at "
                          f"t={t:.0f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
                next_replan = t + PLAN_WINDOW_S

            try:
                u = active.find_u(x_hat=x_hat, sens=sens, est_sat=est_sat,
                                  os_hat=os_gnc, goal=goal)
            except Exception:
                if is_planner and active is not fallback:
                    n_fallbacks += 1
                    n_track_fallbacks += 1
                    active = fallback
                    u = active.find_u(x_hat=x_hat, sens=sens, est_sat=est_sat,
                                      os_hat=os_gnc, goal=goal)
                else:
                    raise
            u = np.asarray(u, float)
            # Hardware envelope, true state, every controller including the planner.
            if n_rw:
                u = enforce_wheel_envelope(u, x[7:7 + n_rw], _hmax_chk, n_mtq, n_rw, dt)

            if is_planner and active is controller:
                atraj = getattr(controller, "active_trajectory", None)
                if atraj is not None and atraj.is_valid_time(os_gnc.J2000):
                    x_ref = np.ravel(atraj.get_state_at(os_gnc.J2000))
                    if x_ref.size >= 7:
                        q_ref = x_ref[3:7]
                        nrm = float(np.linalg.norm(q_ref))
                        if nrm > 1e-9:
                            q_r = q_ref / nrm
                            dq = quat_mult(quat_inv(q_r), x[3:7])
                            if dq[0] < 0:
                                dq = -dq
                            e_vec = 2.0 * dq[1:]          # small-angle error, body frame
                            plan_dev_hist[i] = 2.0 * np.arccos(min(1.0, abs(float(dq[0]))))
                            if bn > 0:
                                b_hat = B_b / bn
                                e_B = float(e_vec @ b_hat)
                                plan_dev_alongB_hist[i] = abs(e_B)
                                plan_dev_perpB_hist[i] = float(
                                    np.linalg.norm(e_vec - e_B * b_hat))

            a_ = getattr(controller, "last_alpha", None)
            if a_ is not None:
                alpha_hist[i] = float(a_)

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
            "eclipse": eclipse_hist, "nadir_eci": nadir_hist,
            "sigma": sigma_hist, "alpha": alpha_hist, "h_frac": hfrac_hist,
            "B_mag": Bmag_hist, "omega": omega_hist,
            "kd": float(getattr(controller, "d_gain", np.nan)),
            "m_max": m_max, "h_max": h_max_eff,
            "B_body0": B_body0,
            "n_plans": n_plans, "n_fallbacks": n_fallbacks,
            "n_budget_kills": n_budget_kills, "n_solve_failures": n_solve_failures,
            "n_track_fallbacks": n_track_fallbacks,
            "plan_deviation": plan_dev_hist,
            "plan_dev_alongB": plan_dev_alongB_hist,
            "plan_dev_perpB": plan_dev_perpB_hist,
            "n_mtq": n_mtq, "m_max": m_max,
        }
    finally:
        release_worker_slot(slot_id)


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------

def boresight_knowledge_series(run: Dict[str, Any]) -> np.ndarray:
    """Knowledge error PROJECTED ONTO THE BORESIGHT [deg] -- commensurable with pointing.

    The full 3-axis attitude error ``2 arccos|q_hat . q|`` is **not** comparable with a
    boresight pointing error: it counts the roll component, which is exactly where a star
    tracker is ~6x worse and where gyro roll drift accumulates, and which does not move the
    boresight at all. Comparing them directly made pointing look 30% *better* than knowledge
    in the one-tracker cell, which is impossible for a loop closed on the estimate.

    This returns the angle between the estimated and true boresight directions in ECI, which
    is the part of the knowledge error the pointing metric can actually see.
    """
    b = np.asarray(IAC_6U.boresight, float)
    q_t = run["state"][:, 3:7]
    q_e = run["est"][:, 3:7]

    def _rot(q, v):
        w, xyz = q[:, 0], q[:, 1:]
        t = 2.0 * np.cross(xyz, v)
        return v + w[:, None] * t + np.cross(xyz, t)

    bt, be = _rot(q_t, b), _rot(q_e, b)
    nt = np.linalg.norm(bt, axis=1); ne = np.linalg.norm(be, axis=1)
    ok = np.isfinite(ne) & (ne > 0) & (nt > 0)
    out = np.full(q_t.shape[0], np.nan)
    cos = np.sum(bt[ok] * be[ok], axis=1) / (nt[ok] * ne[ok])
    out[ok] = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def error_series(run: Dict[str, Any]) -> np.ndarray:
    """Pointing error [deg]: 3-axis attitude error for 'full', boresight angle for 'reduced'."""
    cfg = run["config"]
    q = run["state"][:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if cfg["task"] == "full":
        qr = normalize(np.asarray(cfg["goal_quat"], float))
        return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(q @ qr), 0.0, 1.0)))

    # Reduced attitude: angle between the body boresight in ECI and the target direction.
    # For a nadir-staring task the target MOVES -- it is the instantaneous nadir, not a fixed
    # inertial vector. Scoring nadir-staring against cfg["goal_vec"] compares the boresight
    # with a random fixed direction and reports ~69 deg for a spacecraft that is tracking
    # perfectly.
    b = np.asarray(IAC_6U.boresight, float)
    w, xq, yq, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Rotate the body boresight into ECI:  v = q (x) b (x) q*
    t = 2.0 * np.cross(np.column_stack([xq, yq, zq]), b)
    b_eci = b + w[:, None] * t + np.cross(np.column_stack([xq, yq, zq]), t)

    if cfg["task"] == "nadir":
        tgt = run["nadir_eci"][:q.shape[0]]
        cosang = np.clip(np.sum(b_eci * tgt, axis=1), -1.0, 1.0)
    else:
        tgt = normalize(np.asarray(cfg["goal_vec"], float))
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
    # Geometry covariates, per trial. Logged so the divergence correlation is answerable
    # from this dataset rather than from a second campaign.
    sig_med, sig_min, sig_dwell = [], [], []
    alp_med, alp_min, alp_lowfrac = [], [], []
    h_max_t, h_end_t, ang_eB, along_a = [], [], [], []
    # --- the DISCRIMINATING covariates: damping saturation vs frontier geometry ---------
    # Damping saturation predicts failure correlated with PEAK rate (which high kp generates
    # regardless of the initial draw), independent of goal geometry. The frontier predicts
    # failure correlated with along-field goal geometry, independent of rate. One run, both
    # hypotheses.
    #
    # The unifying inequality both ceilings obey: gyroscopic omega x h is perpendicular to
    # omega and damping kd*omega opposes it, so they add in quadrature and only the
    # transverse magnetorquers can serve either:
    #
    #     sqrt(h^2 + kd^2) * omega  <~  tau_perp
    #
    # One design rule covering Campaign C's variable and this sweep's failure mode.
    pk_omega, damp_ratio, quad_ratio, alpha_lag = [], [], [], []
    fb_frac = []
    fb_split = []
    despin = []
    plan_dev_med, plan_dev_max = [], []
    plan_dev_alongB_energy = []
    est_bore_med, est_bore_p95 = [], []

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

        sg = r.get("sigma")
        if sg is not None:
            sg = sg[:k][np.isfinite(sg[:k])]
            if sg.size:
                sig_med.append(float(np.median(sg))); sig_min.append(float(sg.min()))
                sig_dwell.append(float(np.mean(sg < 0.2)))
        al = r.get("alpha")
        if al is not None:
            al = al[:k][np.isfinite(al[:k])]
            if al.size:
                alp_med.append(float(np.median(al))); alp_min.append(float(al.min()))
                alp_lowfrac.append(float(np.mean(al < 0.5)))
        pd_ = r.get("plan_deviation")
        if pd_ is not None:
            pd_f = pd_[:k][np.isfinite(pd_[:k])]
            if pd_f.size:
                plan_dev_med.append(float(np.degrees(np.median(pd_f))))
                plan_dev_max.append(float(np.degrees(np.max(pd_f))))
                eB = r.get("plan_dev_alongB"); eP = r.get("plan_dev_perpB")
                if eB is not None and eP is not None:
                    b2 = np.nansum(eB[:k] ** 2); p2 = np.nansum(eP[:k] ** 2)
                    if b2 + p2 > 0:
                        # Energy fraction along B. Isotropy baseline ~1/3 (one axis of
                        # three); >>1/3 = deviation concentrated where the magnetorquers
                        # cannot act.
                        plan_dev_alongB_energy.append(float(b2 / (b2 + p2)))

        # Desat sign trace: does the loop ever try to despin? hdot = -u, so despin means
        # u and h share a sign. A low fraction with h climbing = the desat channel starved
        # (the alpha-collapse pathway); a high fraction with h climbing anyway = despin
        # commanded but priced out of the LP box by the dump-blind geometry.
        uh = r.get("u"); st = r.get("state")
        if uh is not None and st is not None and r["config"]["n_rw"]:
            nm_, nr_ = r.get("n_mtq", 3), r["config"]["n_rw"]
            uw = uh[:k, nm_:nm_ + nr_]; hw = st[:k, 7:7 + nr_]
            act = np.abs(uw) > 1e-9
            if act.any():
                despin.append(float(np.mean((uw * hw > 0)[act])))

        npl, nfb = r.get("n_plans", 0), r.get("n_fallbacks", 0)
        if npl + nfb > 0:
            fb_frac.append(float(nfb / (npl + nfb)))
            # Separate columns by construction (see _PlanBudgetKill): a budget-kill
            # and a non-convergence read in OPPOSITE directions.
            fb_split.append({"seed": int(r["config"]["seed"]),
                             "budget_kills": int(r.get("n_budget_kills", 0)),
                             "solve_failures": int(r.get("n_solve_failures", 0)),
                             "track_fallbacks": int(r.get("n_track_fallbacks", 0))})

        om = r.get("omega"); Bm = r.get("B_mag")
        if om is not None and Bm is not None and om[:k].size:
            w = om[:k]
            pk = float(np.max(w))
            pk_omega.append(pk)
            tau_perp = float(np.sqrt(2.0) * r.get("m_max", 0.2) * np.median(Bm[:k]))
            kd = float(r.get("kd", np.nan))
            h_store = float(r.get("h_max", IAC_6U.h_max)) * (
                float(np.median(r["h_frac"][:k])) if r.get("h_frac") is not None else 0.0)
            if tau_perp > 0 and np.isfinite(kd):
                damp_ratio.append(kd * pk / tau_perp)
                quad_ratio.append(float(np.hypot(h_store, kd)) * pk / tau_perp)
            # alpha lead-lag: does alpha collapse PRECEDE rate growth (mechanism) or follow
            # it (detector)? Positive lag = alpha leads.
            al = r.get("alpha")
            if al is not None and al[:k].size > 200:
                a = np.nan_to_num(al[:k], nan=1.0)
                x1 = -(a - np.mean(a)); x2 = w - np.mean(w)
                if x1.std() > 0 and x2.std() > 0:
                    # corr( x1[t - d], x2[t] ) over the overlap; d > 0 means alpha LEADS.
                    L, n_ = 120, len(x1)
                    cc = []
                    for d in range(-L, L + 1):
                        i0, i1 = max(0, d), min(n_, n_ + d)     # x1 indices shifted by d
                        a1 = x1[i0 - d: i1 - d] if d <= 0 else x1[: n_ - d]
                        a2 = x2[i0: i1] if d <= 0 else x2[d:]
                        m_ = min(len(a1), len(a2))
                        cc.append(np.corrcoef(a1[:m_], a2[:m_])[0, 1] if m_ > 50 else np.nan)
                    alpha_lag.append(float(np.arange(-L, L + 1)[int(np.nanargmax(cc))]))

        hf = r.get("h_frac")
        if hf is not None and hf[:k].size:
            h_max_t.append(float(np.max(hf[:k]))); h_end_t.append(float(hf[:k][-1]))

        # Initial geometry: where the required correction sits relative to B and the wheel.
        try:
            cfg0 = r["config"]
            C0 = quat_to_rot(np.asarray(cfg0["q0"], float))
            a_hat = np.asarray(IAC_6U.boresight, float)
            if cfg0["task"] == "full":
                qg = normalize(np.asarray(cfg0["goal_quat"], float))
                dq = quat_mult(quat_inv(np.asarray(cfg0["q0"], float)), qg)
                ax = normalize(dq[1:]) if np.linalg.norm(dq[1:]) > 1e-9 else a_hat
            else:
                tgt = normalize(np.asarray(cfg0["goal_vec"], float))
                v = np.cross(C0 @ a_hat, tgt)
                ax = normalize(C0.T @ v) if np.linalg.norm(v) > 1e-9 else a_hat
            along_a.append(float(abs(ax @ a_hat)))
            B0 = np.ravel(r.get("B_body0", np.zeros(3)))[:3]
            if np.linalg.norm(B0) > 0:
                ang_eB.append(float(np.degrees(np.arccos(np.clip(
                    abs(ax @ B0 / np.linalg.norm(B0)), 0.0, 1.0)))))
        except Exception:
            pass
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
        # Boresight-projected knowledge: the only version comparable with pointing error.
        try:
            bk = boresight_knowledge_series(r)[h0:k]
            bk = bk[np.isfinite(bk)]
            if bk.size:
                est_bore_med.append(float(np.median(bk)))
                est_bore_p95.append(float(np.percentile(bk, 95)))
        except Exception:
            pass

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
    _task = runs[0]["config"]["task"] if runs and runs[0] else "reduced"

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
        # Knowledge matched to the GOAL TYPE -- this is the figure to use in any
        # knowledge-vs-control decomposition, and which one is correct depends on the cell:
        #
        #   reduced-attitude goals: roll is unconstrained and invisible to the pointing
        #       metric, so boresight-projected knowledge is the commensurable one.
        #   full-attitude goals:    roll is part of the objective, so 3-axis knowledge is
        #       correct. Using the projected figure here would understate knowledge error in
        #       exactly the axis where the star tracker is ~6x worse.
        #
        # `matched_knowledge_deg` below selects automatically; state which in the caption.
        "median_bore_knowledge_deg": _nanmed(est_bore_med) if est_bore_med else None,
        "p95_bore_knowledge_deg": _nanmed(est_bore_p95) if est_bore_p95 else None,
        "matched_knowledge_deg": (
            (_nanmed(est_med) if est_med else None) if _task == "full"
            else (_nanmed(est_bore_med) if est_bore_med else None)),
        "matched_knowledge_p95_deg": (
            (_nanmed(est_p95) if est_p95 else None) if _task == "full"
            else (_nanmed(est_bore_p95) if est_bore_p95 else None)),
        "knowledge_basis": ("3-axis (full-attitude goal)" if _task == "full"
                            else "boresight-projected (reduced/nadir goal)"),
        # Fraction of the original residual dipole still standing after cancellation, and
        # how much of that leftover is secular (the part a body-fixed wheel integrates)
        # rather than cyclic.
        "median_dipole_residual_frac": _nanmed(dip_frac) if dip_frac else None,
        "median_dipole_residual_secular_frac": _nanmed(dip_sec_frac) if dip_sec_frac else None,
        "finals_deg": finals.tolist(),
        # Per-trial, so error-vs-availability and error-vs-eclipse correlations can be done
        # from disk without re-running anything.
        "per_trial_seed": [int(r["config"]["seed"]) for r in runs if r is not None],
        "per_trial_tracker_avail": avail,
        "per_trial_eclipse_frac": ecl,
        "per_trial_est_att_err_deg": est_med,
        # --- geometry covariates for the divergence correlation ---
        "per_trial_sigma_median": sig_med,
        "per_trial_sigma_min": sig_min,
        "per_trial_sigma_dwell_below_0p2": sig_dwell,
        "per_trial_alpha_median": alp_med,
        "per_trial_alpha_min": alp_min,
        "per_trial_alpha_frac_below_0p5": alp_lowfrac,
        "per_trial_h_frac_max": h_max_t,
        "per_trial_h_frac_end": h_end_t,
        # NOTE: for reduced-attitude / nadir goals this is identically ZERO, and that is a
        # structural fact rather than a logging bug. A boresight-pointing correction is a
        # rotation about an axis perpendicular to the boresight (v = b_hat x t_hat), and the
        # wheel is mounted ALONG the boresight -- so the wheel's single torque axis is
        # orthogonal to every correction the task ever requires. It contributes nothing
        # directly to boresight pointing; all of it comes from the rank-2 magnetorquers, and
        # the wheel helps only indirectly (roll damping, momentum management, gyroscopic
        # coupling). Informative for FULL-attitude goals, where roll is part of the objective.
        #
        # This makes Campaign D's mounting question three-way rather than two-way:
        # boresight mounting maximises sigma (restoration duty) and has the worst dump
        # margin, AND puts the wheel's torque axis where the pointing objective cannot use it.
        "per_trial_along_a_share": along_a,
        "per_trial_err_axis_to_B_deg": ang_eB,
        # --- damping-vs-geometry discriminators ---
        "per_trial_peak_omega_dps": [float(np.degrees(v)) for v in pk_omega],
        "per_trial_damping_ratio": damp_ratio,      # kd*|w|_peak / tau_perp ; >1 = saturated
        "per_trial_quadrature_ratio": quad_ratio,   # sqrt(h^2+kd^2)*|w|_peak / tau_perp
        "per_trial_alpha_lead_lag_s": alpha_lag,    # >0 = alpha collapse LEADS rate growth
        # Planner cells only: fraction of replanning windows that fell back to reactive PD.
        # A planner cell executing PD through its hard windows inherits PD's behaviour in
        # exactly the trials where the two differ -- biasing the comparison toward "no
        # difference" where it matters most. A cell at high fallback fraction is not a
        # planner measurement and must be reported as such.
        "per_trial_fallback_frac": fb_frac,
        "mean_fallback_frac": float(np.mean(fb_frac)) if fb_frac else None,
        "per_trial_fallback_split": fb_split,
        "budget_kill_seeds": sorted(d["seed"] for d in fb_split if d["budget_kills"]),
        "solve_failure_seeds": sorted(d["seed"] for d in fb_split if d["solve_failures"]),
        "total_budget_kills": int(sum(d["budget_kills"] for d in fb_split)),
        "total_solve_failures": int(sum(d["solve_failures"] for d in fb_split)),
        "total_track_fallbacks": int(sum(d["track_fallbacks"] for d in fb_split)),
        # Plan-vs-executed attitude divergence [deg]: small max + bad final = the PLAN missed;
        # large max = the TRACKER lost it. The discriminator for planner-cell failures.
        "per_trial_plan_dev_median_deg": plan_dev_med,
        "per_trial_plan_dev_max_deg": plan_dev_max,
        "per_trial_plan_dev_alongB_energy_frac": plan_dev_alongB_energy,
        "per_trial_despin_frac": despin,
    }

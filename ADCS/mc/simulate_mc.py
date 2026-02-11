from __future__ import annotations

__all__ = ["simulate_mc"]

import os
import sys
import pickle
import hashlib
import inspect
import numpy as np
from typing import Optional, Any, Dict, List, Union
from contextlib import contextmanager
from scipy.integrate import solve_ivp

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller, PlanAndTrackBase
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.simresults import SimulationResults, RunResults
from ADCS.mc.monte_carlo_runner import MonteCarloRunner, claim_worker_slot, release_worker_slot, update_worker_progress

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ADCS.mc.mcconfig import MCConfig


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _picklable(name: str, obj: Any) -> None:
    try:
        pickle.dumps(obj)
    except Exception as e:
        raise TypeError(f"{name} is not picklable: {e}") from e


def _is_sampler(v: Any) -> bool:
    return callable(v)


def _sample(v: Any, rng: np.random.Generator) -> Any:
    if not _is_sampler(v):
        return v
    try:
        sig = inspect.signature(v)
        if len(sig.parameters) == 0:
            return v()
        return v(rng)
    except Exception:
        try:
            return v(rng)
        except TypeError:
            return v()


def _as_1d_float(x: Any, n: Optional[int], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and a.size != n:
        raise ValueError(f"{name} must have length {n}, got {a.size}")
    return a


def _freeze_os0(os0: Orbital_State) -> Dict[str, Any]:
    return os0.to_dict()


def _thaw_os0(payload: Dict[str, Any], ephem: Ephemeris) -> Orbital_State:
    return Orbital_State.from_dict(payload, ephem=ephem, density_model=None, fast=True)


def _freeze_os_hist(hist: Any) -> Any:
    if hist is None:
        return None
    return [os_i.to_dict() if os_i is not None else None for os_i in hist]


def _thaw_os_hist(hist: Any, ephem: Ephemeris) -> Any:
    if hist is None:
        return None
    return [
        Orbital_State.from_dict(d, ephem=ephem, density_model=None, fast=True) if d is not None else None
        for d in hist
    ]


def _orbit_seq_from_orbit_obj(orb: Orbit, start_J2000: float, dt: float, tf: float) -> List[Dict[str, Any]]:
    sec2cent = TimeConstants.sec2cent
    N = int(tf / dt)
    out: List[Dict[str, Any]] = []
    for k in range(N + 1):
        out.append(orb.get_os(J2000=start_J2000 + k * dt * sec2cent).to_dict())
    return out


def _orbit_seq_from_os0(os0: Orbital_State, dt: float, tf: float, use_J2: bool, fast: bool) -> List[Orbital_State]:
    sec2cent = TimeConstants.sec2cent
    N = int(tf / dt)
    end_time = os0.J2000 + tf * sec2cent
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=use_J2, fast=fast, verbose=False)
    out: List[Orbital_State] = []
    for k in range(N + 1):
        out.append(orb.get_os(J2000=os0.J2000 + k * dt * sec2cent))
    return out


def _simulate_with_precomputed_orbit(
    *,
    x: np.ndarray,
    satellite: Satellite,
    est_satellite: Optional[EstimatedSatellite],
    controller: Optional[Controller],
    estimator: Optional[Attitude_Estimator],
    orbit_estimator: Optional[Orbit_Estimator],
    goal: Optional[Union[Goal, GoalList]],
    os_seq: List[Orbital_State],
    dt: float,
    tf: float,
    slot_id: int = -1,  # Added for UI tracking
    run_id: int = -1,   # Added for UI tracking
) -> RunResults:
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length {satellite.state_len}. "
            f"It must be 7 + N_rw."
        )

    N = int(tf / dt)
    if len(os_seq) < N + 1:
        raise ValueError(f"precomputed orbit must have at least {N+1} states, got {len(os_seq)}")

    os0 = os_seq[0]
    if goal is None:
        goal_list = GoalList({os0.J2000: No_Goal()})
    elif isinstance(goal, Goal):
        goal_list = GoalList({os0.J2000: goal})
    elif isinstance(goal, GoalList):
        goal_list = goal
    else:
        raise ValueError("goal must be None, a Goal, or a GoalList.")

    u = np.zeros(satellite.control_len)

    need_est_sat = (estimator is not None) or (controller is not None)
    if need_est_sat and est_satellite is None:
        est_satellite = EstimatedSatellite.from_satellite(satellite)

    x_hat = None
    if estimator is not None and est_satellite is not None:
        x_hat = np.empty(est_satellite.state_len)

    os_hat = None

    if controller is not None and isinstance(controller, PlanAndTrackBase):
        trajectory = controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=tf,
            x_0=x,
            os_0=os0,
            goals=goal_list,
            verbose=False,
        )
        controller.set_active_trajectory(trajectory)

    run_results = RunResults(satellite=satellite, est_satellite=est_satellite)

    for k in range(N):
        # --- UI UPDATE ---
        if k % 10 == 0 and slot_id != -1:
            update_worker_progress(slot_id, run_id, k, N)

        os_k = os_seq[k]
        os_kp1 = os_seq[k + 1]
        J2000_k = os_k.J2000

        y = satellite.sensor_readings(x=x, os=os_k)
        y_clean = satellite.noiseless_sensor_readings(x=x, os=os_k)

        if orbit_estimator is not None:
            gps = satellite.GPS_readings(x=x, os=os_k)
            os_hat = orbit_estimator.update(GPS_measurements=gps, J2000=J2000_k)
            os_for_gnc = os_hat if os_hat is not None else os_k
        else:
            os_hat = None
            os_for_gnc = os_k

        if estimator is not None:
            x_hat = estimator.update(u=u, sensors=y, os=os_for_gnc)
            x_for_ctrl = x_hat
        else:
            x_for_ctrl = x

        active_goal = goal_list.get_active_goal(J2000_k, time_units="centuries")

        if controller is not None:
            u = controller.find_u(
                x_hat=x_for_ctrl,
                # Use noiseless sensors for control if required by logic, 
                # but standard practice is 'sens' (y)
                sens=y,
                est_sat=est_satellite,
                os_hat=os_for_gnc,
                goal=active_goal,
            )
        else:
            u[:] = 0.0

        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os_k, os_kp1),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        target, w_target = active_goal.to_ref(os_for_gnc)

        est_act_bias_snapshot = None
        est_sens_bias_snapshot = None

        if estimator is not None and x_hat is not None and est_satellite is not None:
            n_rw = int(getattr(est_satellite, "number_RW", 0))
            n_ab = int(getattr(est_satellite, "act_bias_len", 0))
            n_sb = int(getattr(est_satellite, "att_sens_bias_len", 0))

            base = 7 + n_rw
            ab0, ab1 = base, base + n_ab
            sb0, sb1 = ab1, ab1 + n_sb

            if len(x_hat) >= sb1:
                b_act_hat = np.asarray(x_hat[ab0:ab1], dtype=float).reshape(-1)
                b_sens_hat = np.asarray(x_hat[sb0:sb1], dtype=float).reshape(-1)

                act_parts = []
                ai = 0
                for act in getattr(satellite, "actuators", []) or []:
                    if hasattr(act, "bias") and bool(act.bias):
                        dim = int(np.atleast_1d(act.bias.bias).size)
                        act_parts.append(b_act_hat[ai:ai + dim].copy())
                        ai += dim
                    else:
                        act_parts.append(None)

                if act_parts:
                    est_act_bias_snapshot = np.array(act_parts, dtype=object)

                sens_parts = []
                si = 0
                for sens in getattr(satellite, "sensors", []) or []:
                    if hasattr(sens, "bias") and bool(sens.bias):
                        dim = int(np.atleast_1d(sens.bias.bias).size)
                        sens_parts.append(b_sens_hat[si:si + dim].copy())
                        si += dim
                    else:
                        sens_parts.append(None)

                if sens_parts:
                    est_sens_bias_snapshot = np.array(sens_parts, dtype=object)

        run_results.record(
            k=k,
            time_J2000=J2000_k,
            time_s=k * dt,
            os=os_k,
            est_os=os_hat,
            os_cov=(getattr(getattr(orbit_estimator, "os_hat", None), "P", None) if orbit_estimator is not None else None),
            state=x,
            est_state=x_hat,
            state_cov=(getattr(getattr(estimator, "x_hat", None), "cov", None) if estimator is not None else None),
            actuator_bias=(
                np.array([np.atleast_1d(act.bias.bias) for act in satellite.actuators], dtype=object)
                if getattr(satellite, "actuators", None) else None
            ),
            sensor_bias=(
                np.array([np.atleast_1d(sens.bias.bias) for sens in satellite.sensors], dtype=object)
                if getattr(satellite, "sensors", None) else None
            ),
            est_actuator_bias=est_act_bias_snapshot,
            est_sensor_bias=est_sens_bias_snapshot,
            target=target,
            w_target=w_target,
            clean_sensor=y_clean,
            sensor=y,
            control=u,
        )

    return run_results


_EPHEM: Optional[Ephemeris] = None
_ORBIT_CACHE: Dict[str, List[Orbital_State]] = {}


def _get_ephem() -> Ephemeris:
    global _EPHEM
    if _EPHEM is None:
        _EPHEM = Ephemeris()
    return _EPHEM


def _cache_key(*, os0_payload: Dict[str, Any], dt: float, tf: float, use_J2: bool, fast: bool, slot_id: int) -> str:
    b = pickle.dumps((slot_id, os0_payload, float(dt), float(tf), bool(use_J2), bool(fast)))
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def _simulate_mc_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    slot_id = claim_worker_slot()
    run_id = int(cfg["run_id"])
    seed = int(cfg["seed"])

    try:
        np.random.seed(seed)
        # Initial progress signal
        update_worker_progress(slot_id, run_id, 0, 1)

        ephem = _get_ephem()

        x0 = np.asarray(cfg["x0"], dtype=float).copy()
        satellite: Satellite = cfg["satellite"]
        est_satellite = cfg.get("est_satellite")
        controller = cfg.get("controller")
        estimator = cfg.get("estimator")
        orbit_estimator = cfg.get("orbit_estimator")
        goal = cfg.get("goal")
        dt = float(cfg["dt"])
        tf = float(cfg["tf"])

        orbit_mode = cfg.get("orbit_mode", "os0")
        use_J2 = bool(cfg.get("orbit_use_J2", True))
        fast = bool(cfg.get("orbit_fast", False))

        if orbit_mode == "seq":
            os_seq_payload = cfg.get("orbit_os_seq")
            if os_seq_payload is None:
                raise ValueError("orbit_mode='seq' requires orbit_os_seq")
            os_seq = _thaw_os_hist(os_seq_payload, ephem=ephem)
        else:
            os0_payload = cfg["os0_payload"]
            key = _cache_key(os0_payload=os0_payload, dt=dt, tf=tf, use_J2=use_J2, fast=fast, slot_id=slot_id)
            os_seq = _ORBIT_CACHE.get(key)
            if os_seq is None:
                os0 = _thaw_os0(os0_payload, ephem=ephem)
                os_seq = _orbit_seq_from_os0(os0=os0, dt=dt, tf=tf, use_J2=use_J2, fast=fast)
                _ORBIT_CACHE[key] = os_seq

        with suppress_output():
            sim_results = _simulate_with_precomputed_orbit(
                x=x0,
                satellite=satellite,
                est_satellite=est_satellite,
                controller=controller,
                estimator=estimator,
                orbit_estimator=orbit_estimator,
                goal=goal,
                os_seq=os_seq,
                dt=dt,
                tf=tf,
                slot_id=slot_id, # Passed for tracking
                run_id=run_id,   # Passed for tracking
            )

        sim_results.os_hist = _freeze_os_hist(sim_results.os_hist)
        sim_results.est_os_hist = _freeze_os_hist(sim_results.est_os_hist)

        # Final progress signal
        update_worker_progress(slot_id, run_id, 1, 1)

        return {
            "run_id": run_id,
            "seed": int(cfg["seed"]),
            "applied": cfg.get("applied", {}),
            "results": sim_results,
        }

    finally:
        release_worker_slot(slot_id)


def simulate_mc(
    x: np.ndarray,
    satellite: Satellite,
    est_satellite: Optional[EstimatedSatellite] = None,
    controller: Optional[Controller] = None,
    estimator: Optional[Attitude_Estimator] = None,
    orbit_estimator: Optional[Orbit_Estimator] = None,
    goal: Optional[Goal | GoalList] = None,
    os0: Orbital_State = None,
    dt: float = 1.0,
    tf: float = 500.0,
    mc_config: Optional["MCConfig"] = None,
    num_runs: int = 100,
    max_workers: Optional[int] = None,
    base_seed: int = 0,
) -> SimulationResults:
    r"""
    Run a Monte Carlo ensemble of ADCS simulations using parallel workers.

    This function generates ``num_runs`` independent simulation configurations by
    applying optional per-run sampling rules from :class:`~ADCS.mc.mcconfig.MCConfig`,
    then executes each run via :class:`~ADCS.mc.monte_carlo_runner.MonteCarloRunner`.
    Each run simulates spacecraft attitude dynamics, sensors, estimation, and control,
    while optionally sampling initial conditions and run parameters such as ``dt``,
    ``tf``, angular rate, quaternion, reaction wheel momentum, goals, and orbit
    overrides.

    Orbit handling supports two modes:

    +------------+---------------------------------------------------------------+
    | Mode       | Description                                                   |
    +------------+---------------------------------------------------------------+
    | os0        | Propagate an :class:`~ADCS.orbits.orbital_state.Orbital_State`|
    |            | forward using :class:`~ADCS.orbits.orbit.Orbit`.              |
    +------------+---------------------------------------------------------------+
    | seq        | Use a precomputed orbital-state sequence (for example from an |
    |            | :class:`~ADCS.orbits.orbit.Orbit` override or an explicit     |
    |            | list).                                                        |
    +------------+---------------------------------------------------------------+

    For multiprocessing compatibility, all provided models (satellite, controller,
    estimators, goal objects) must be picklable.

    :param x:
        Base initial true satellite state vector. The length must match
        ``satellite.state_len`` and is expected to follow the satellite state
        convention (angular velocity, quaternion, reaction wheel states, etc.).
        Per-run overrides may be applied by ``mc_config``.
    :type x:
        numpy.ndarray

    :param satellite:
        The true satellite model, including dynamics, sensors, and actuators.
        This object must be picklable for parallel execution.
    :type satellite:
        :class:`~ADCS.satellite_hardware.satellite.Satellite`

    :param est_satellite:
        Estimated satellite model used by estimators and controllers. If ``None``,
        it may be constructed internally on a per-run basis when required by the
        provided estimator or controller. If provided, it must be picklable.
    :type est_satellite:
        :class:`~ADCS.satellite_hardware.satellite.EstimatedSatellite` or None

    :param controller:
        Control law used to compute actuator commands. If the controller is a
        :class:`~ADCS.controller.PlanAndTrackBase`, an initial trajectory is computed
        at the start of each run based on the run-specific initial conditions and
        goals. If provided, it must be picklable.
    :type controller:
        :class:`~ADCS.controller.Controller` or None

    :param estimator:
        Attitude estimator used to estimate spacecraft state from sensor
        measurements. If provided, it must be picklable.
    :type estimator:
        :class:`~ADCS.estimators.attitude_estimators.Attitude_Estimator` or None

    :param orbit_estimator:
        Orbit estimator used to estimate orbital state from GPS measurements.
        If provided, it must be picklable.
    :type orbit_estimator:
        :class:`~ADCS.estimators.orbit_estimators.Orbit_Estimator` or None

    :param goal:
        Desired attitude or pointing objective. This may be ``None`` (no goal),
        a single :class:`~ADCS.CONOPS.goals.Goal`, or a
        :class:`~ADCS.CONOPS.goallist.GoalList` defining time-varying goals.
        Per-run overrides may be applied by ``mc_config``. If provided, it must be
        picklable.
    :type goal:
        :class:`~ADCS.CONOPS.goals.Goal`,
        :class:`~ADCS.CONOPS.goallist.GoalList`,
        or None

    :param os0:
        Initial orbital state at the start of the Monte Carlo campaign. This must
        be provided and is used as the base orbit for propagation in runs that do
        not override the orbit.
    :type os0:
        :class:`~ADCS.orbits.orbital_state.Orbital_State`

    :param dt:
        Base simulation time step in seconds. Per-run overrides may be applied by
        ``mc_config``.
    :type dt:
        float

    :param tf:
        Base simulation duration in seconds. Per-run overrides may be applied by
        ``mc_config``.
    :type tf:
        float

    :param mc_config:
        Monte Carlo configuration describing run-to-run sampling rules for selected
        parameters, such as ``dt``, ``tf``, initial angular velocity, quaternion,
        reaction wheel momentum, goal selection, and orbit overrides.
    :type mc_config:
        :class:`~ADCS.mc.mcconfig.MCConfig` or None

    :param num_runs:
        Number of Monte Carlo runs to execute.
    :type num_runs:
        int

    :param max_workers:
        Maximum number of worker processes to use. If ``None``, the runner chooses
        an implementation-defined default.
    :type max_workers:
        int or None

    :param base_seed:
        Base seed used to deterministically generate per-run seeds as
        ``base_seed + run_id``.
    :type base_seed:
        int

    :return:
        Aggregated results for all completed runs, including a list of per-run
        :class:`~ADCS.helpers.simresults.RunResults`, optional per-run configuration
        summaries, and run identifiers.
    :rtype:
        :class:`~ADCS.helpers.simresults.SimulationResults`

    """
    if os0 is None:
        raise ValueError("os0 must be provided to simulate_mc().")
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length {satellite.state_len}. "
            f"It must be 7 + N_rw."
        )

    _picklable("satellite", satellite)
    if est_satellite is not None:
        _picklable("est_satellite", est_satellite)
    if controller is not None:
        _picklable("controller", controller)
    if estimator is not None:
        _picklable("estimator", estimator)
    if orbit_estimator is not None:
        _picklable("orbit_estimator", orbit_estimator)
    if goal is not None:
        _picklable("goal", goal)

    x_base = np.asarray(x, dtype=float).copy()
    os0_base_payload = _freeze_os0(os0)

    def _build_run_cfg(run_id: int) -> Dict[str, Any]:
        seed = int(base_seed) + int(run_id)
        rng = np.random.default_rng(seed)

        dt_i = float(dt)
        tf_i = float(tf)
        x0_i = x_base.copy()
        goal_i: Optional[Union[Goal, GoalList]] = goal
        applied: Dict[str, Any] = {}

        if mc_config is not None:
            v = getattr(mc_config, "dt", None)
            if v is not None:
                dt_i = float(_sample(v, rng))
                applied["dt"] = dt_i

            v = getattr(mc_config, "tf", None)
            if v is not None:
                tf_i = float(_sample(v, rng))
                applied["tf"] = tf_i

            v = getattr(mc_config, "w", None)
            if v is not None:
                w = _as_1d_float(_sample(v, rng), 3, "mc_config.w")
                x0_i[:3] = w
                applied["w"] = w

            v = getattr(mc_config, "q", None)
            if v is not None:
                q = _as_1d_float(_sample(v, rng), 4, "mc_config.q")
                x0_i[3:7] = q
                applied["q"] = q

            v = getattr(mc_config, "h", None)
            if v is not None:
                h = _as_1d_float(_sample(v, rng), len(x0_i) - 7, "mc_config.h")
                x0_i[7:] = h
                applied["h"] = h

            v = getattr(mc_config, "goal", None)
            if v is not None:
                goal_i = _sample(v, rng)
                applied["goal"] = type(goal_i).__name__ if goal_i is not None else None

        if goal_i is not None:
            _picklable("goal(run)", goal_i)

        orbit_mode = "os0"
        os0_payload = os0_base_payload
        orbit_os_seq: Optional[List[Dict[str, Any]]] = None

        orbit_override = None
        if mc_config is not None:
            orbit_override = getattr(mc_config, "orbit", None)
            if orbit_override is None:
                orbit_override = getattr(mc_config, "os0", None)

        if orbit_override is not None:
            ov = _sample(orbit_override, rng)
            if isinstance(ov, Orbit):
                with suppress_output():
                    orbit_os_seq = _orbit_seq_from_orbit_obj(ov, start_J2000=float(os0_payload["J2000"]), dt=dt_i, tf=tf_i)
                orbit_mode = "seq"
            elif isinstance(ov, list):
                if len(ov) == 0:
                    raise ValueError("orbit override list is empty")
                if isinstance(ov[0], dict):
                    orbit_os_seq = ov
                elif isinstance(ov[0], Orbital_State):
                    orbit_os_seq = [os_i.to_dict() for os_i in ov]
                else:
                    raise TypeError("orbit override list must contain dict or Orbital_State")
                orbit_mode = "seq"
            elif isinstance(ov, Orbital_State):
                os0_payload = ov.to_dict()
                orbit_mode = "os0"
            elif isinstance(ov, dict) and "J2000" in ov and "R" in ov and "V" in ov:
                os0_payload = ov
                orbit_mode = "os0"
            else:
                raise TypeError("orbit/os0 override must be Orbit, Orbital_State, list, or Orbital_State dict")

        cfg: Dict[str, Any] = {
            "run_id": int(run_id),
            "seed": seed,
            "applied": applied,
            "x0": x0_i,
            "dt": dt_i,
            "tf": tf_i,
            "goal": goal_i,
            "orbit_mode": orbit_mode,
            "os0_payload": os0_payload,
            "orbit_use_J2": True,
            "orbit_fast": False,
            "satellite": satellite,
            "est_satellite": est_satellite,
            "controller": controller,
            "estimator": estimator,
            "orbit_estimator": orbit_estimator,
        }

        if orbit_mode == "seq":
            cfg["orbit_os_seq"] = orbit_os_seq

        return cfg

    runner = MonteCarloRunner(
        sim_func=_simulate_mc_worker,
        config_generator=_build_run_cfg,
        num_runs=int(num_runs),
        max_workers=max_workers,
    )

    raw = runner.run()
    cleaned: List[Dict[str, Any]] = [
        r for r in raw
        if r is not None and isinstance(r, dict) and r.get("results") is not None
    ]

    ephem = Ephemeris()
    for item in cleaned:
        res = item["results"]
        res.os_hist = _thaw_os_hist(res.os_hist, ephem=ephem)
        res.est_os_hist = _thaw_os_hist(res.est_os_hist, ephem=ephem)

    runs = [entry["results"] for entry in cleaned]
    configs = [entry.get("applied", {}) for entry in cleaned]
    run_ids = [entry.get("run_id") for entry in cleaned]

    return SimulationResults(runs=runs, configs=configs, run_ids=run_ids)
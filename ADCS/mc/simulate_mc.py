__all__ = ["simulate_mc"]

import os
import sys
import pickle
import numpy as np
from typing import Optional, Any, Dict, List, Union, Iterable
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
from ADCS.helpers.simresults import SimulationResults
from ADCS.helpers.simresults_mc import MCSimulationResults

from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)

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


def _is_sampler(v: Any) -> bool:
    return callable(v)


def _sample(v: Any, rng: np.random.Generator) -> Any:
    return v() if _is_sampler(v) else v


def _as_1d_float(x: Any, n: Optional[int], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and a.size != n:
        raise ValueError(f"{name} must have length {n}, got {a.size}")
    return a


def _picklable(name: str, obj: Any) -> None:
    try:
        pickle.dumps(obj)
    except Exception as e:
        raise TypeError(f"{name} is not picklable: {e}") from e


def _freeze_os0(os0: Orbital_State) -> Dict[str, Any]:
    return {
        "J2000": float(os0.J2000),
        "R": np.asarray(os0.R, dtype=float).reshape(3).copy(),
        "V": np.asarray(os0.V, dtype=float).reshape(3).copy(),
        "S": np.asarray(os0.S, dtype=float).reshape(3).copy() if getattr(os0, "S", None) is not None else None,
        "B": np.asarray(os0.B, dtype=float).reshape(3).copy() if getattr(os0, "B", None) is not None else None,
        "rho": float(os0.rho) if getattr(os0, "rho", None) is not None else None,
    }


def _thaw_os0(payload: Dict[str, Any], ephem: Ephemeris) -> Orbital_State:
    return Orbital_State.from_dict(payload, ephem=ephem, density_model=None, fast=True)


def _freeze_os_hist(hist: Any) -> Any:
    if hist is None:
        return None
    out = []
    for os0 in hist:
        out.append(os0.to_dict() if os0 is not None else None)
    return out


def _thaw_os_hist(hist: Any, ephem: Ephemeris) -> Any:
    if hist is None:
        return None
    out = []
    for d in hist:
        out.append(Orbital_State.from_dict(d, ephem=ephem, density_model=None, fast=True) if d is not None else None)
    return out


def _resolve_os0_from_override(os0_base: Orbital_State, override: Any) -> Orbital_State:
    if override is None:
        return os0_base
    if isinstance(override, Orbital_State):
        return override
    if isinstance(override, Orbit):
        return override.get_os(J2000=os0_base.J2000)
    if isinstance(override, dict) and "J2000" in override and "R" in override and "V" in override:
        return Orbital_State.from_dict(override, ephem=Ephemeris(), density_model=None, fast=True)
    raise TypeError("orbit/os0 override must be an Orbit, Orbital_State, or Orbital_State dict")


def _orbit_to_dict_list(
    orbit_obj: Any,
    *,
    os0_base: Orbital_State,
    dt: float,
    tf: float,
    use_J2: bool = True,
    fast: bool = False,
) -> List[Dict[str, Any]]:
    sec2cent = TimeConstants.sec2cent
    N = int(tf / dt)

    if isinstance(orbit_obj, list):
        if len(orbit_obj) == 0:
            raise ValueError("orbit list is empty")
        if isinstance(orbit_obj[0], Orbital_State):
            return [os_i.to_dict() for os_i in orbit_obj]
        if isinstance(orbit_obj[0], dict):
            return orbit_obj
        raise TypeError("orbit list must contain Orbital_State or dict entries")

    if isinstance(orbit_obj, Orbit):
        start_time = os0_base.J2000
        out: List[Dict[str, Any]] = []
        for k in range(N + 1):
            J2000_k = start_time + k * dt * sec2cent
            out.append(orbit_obj.get_os(J2000=J2000_k).to_dict())
        return out

    if isinstance(orbit_obj, Orbital_State):
        start_time = orbit_obj.J2000
        end_time = start_time + tf * sec2cent
        orb = Orbit(os0=orbit_obj, end_time=end_time, dt=dt, use_J2=use_J2, fast=fast)
        out: List[Dict[str, Any]] = []
        for k in range(N + 1):
            J2000_k = start_time + k * dt * sec2cent
            out.append(orb.get_os(J2000=J2000_k).to_dict())
        return out

    if isinstance(orbit_obj, dict) and "J2000" in orbit_obj and "R" in orbit_obj and "V" in orbit_obj:
        os0 = Orbital_State.from_dict(orbit_obj, ephem=Ephemeris(), density_model=None, fast=True)
        return _orbit_to_dict_list(os0, os0_base=os0, dt=dt, tf=tf, use_J2=use_J2, fast=fast)

    raise TypeError("orbit override must be Orbit, Orbital_State, list, or Orbital_State dict")


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
) -> SimulationResults:
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length "
            f"{satellite.state_len}. It must be 7 + N_rw."
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
    if estimator is not None:
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

    sim_results = SimulationResults(satellite=satellite, est_satellite=est_satellite)

    for k in range(N):
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
            n_rw = getattr(est_satellite, "number_RW", 0)
            n_ab = getattr(est_satellite, "act_bias_len", 0)
            n_sb = getattr(est_satellite, "att_sens_bias_len", 0)

            base = 7 + int(n_rw)
            ab0, ab1 = base, base + int(n_ab)
            sb0, sb1 = ab1, ab1 + int(n_sb)

            if len(x_hat) >= sb1:
                b_act_hat = np.asarray(x_hat[ab0:ab1], dtype=float).reshape(-1)
                b_sens_hat = np.asarray(x_hat[sb0:sb1], dtype=float).reshape(-1)

                act_parts = []
                ai = 0
                if getattr(satellite, "actuators", None):
                    for act in satellite.actuators:
                        if hasattr(act, "bias") and bool(act.bias):
                            dim = int(np.atleast_1d(act.bias.bias).size)
                            act_parts.append(
                                b_act_hat[ai:ai + dim].reshape(dim, 1) if dim == 1 else b_act_hat[ai:ai + dim]
                            )
                            ai += dim
                        else:
                            act_parts.append(None)

                if len(act_parts) == 0:
                    est_act_bias_snapshot = None
                else:
                    if ai != b_act_hat.size:
                        est_act_bias_snapshot = np.array([b_act_hat.copy()], dtype=object)
                    else:
                        est_act_bias_snapshot = np.array(act_parts, dtype=object)

                sens_parts = []
                si = 0
                if getattr(satellite, "sensors", None):
                    for sens in satellite.sensors:
                        if hasattr(sens, "bias") and bool(sens.bias):
                            dim = int(np.atleast_1d(sens.bias.bias).size)
                            sens_parts.append(
                                b_sens_hat[si:si + dim].reshape(dim, 1) if dim == 1 else b_sens_hat[si:si + dim]
                            )
                            si += dim
                        else:
                            sens_parts.append(None)

                if len(sens_parts) == 0:
                    est_sens_bias_snapshot = None
                else:
                    if si != b_sens_hat.size:
                        est_sens_bias_snapshot = np.array([b_sens_hat.copy()], dtype=object)
                    else:
                        est_sens_bias_snapshot = np.array(sens_parts, dtype=object)

        sim_results.record(
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

    return sim_results


def _simulate_mc_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    slot_id = claim_worker_slot()
    run_id = int(cfg["run_id"])

    try:
        update_worker_progress(slot_id, run_id, 0, 1)

        ephem = Ephemeris()

        x0 = np.asarray(cfg["x0"], dtype=float).copy()
        satellite: Satellite = cfg["satellite"]
        est_satellite = cfg.get("est_satellite")
        controller = cfg.get("controller")
        estimator = cfg.get("estimator")
        orbit_estimator = cfg.get("orbit_estimator")
        goal = cfg.get("goal")
        dt = float(cfg["dt"])
        tf = float(cfg["tf"])

        os_seq_payload = cfg.get("orbit_os_seq", None)
        if os_seq_payload is None:
            os0 = _thaw_os0(cfg["os0_payload"], ephem=ephem)
            os_seq = _orbit_to_dict_list(os0, os0_base=os0, dt=dt, tf=tf, use_J2=True, fast=False)
            os_seq_payload = os_seq

        os_seq = _thaw_os_hist(os_seq_payload, ephem=ephem)

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
            )

        sim_results.os_hist = _freeze_os_hist(sim_results.os_hist)
        sim_results.est_os_hist = _freeze_os_hist(sim_results.est_os_hist)

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
) -> MCSimulationResults:
    if os0 is None:
        raise ValueError("os0 must be provided to simulate_mc().")
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length "
            f"{satellite.state_len}. It must be 7 + N_rw."
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
    os0_base = os0

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

        orbit_override = None
        if mc_config is not None:
            orbit_override = getattr(mc_config, "orbit", None)
            if orbit_override is None:
                orbit_override = getattr(mc_config, "os0", None)

        if orbit_override is not None:
            ov = _sample(orbit_override, rng)
            os0_for_orbit = _resolve_os0_from_override(os0_base, ov)
        else:
            os0_for_orbit = os0_base

        with suppress_output():
            orbit_os_seq = _orbit_to_dict_list(
                os0_for_orbit,
                os0_base=os0_for_orbit,
                dt=dt_i,
                tf=tf_i,
                use_J2=True,
                fast=False,
            )

        _picklable("goal(run)", goal_i)

        return {
            "run_id": int(run_id),
            "seed": seed,
            "applied": applied,
            "x0": x0_i,
            "dt": dt_i,
            "tf": tf_i,
            "goal": goal_i,
            "orbit_os_seq": orbit_os_seq,
            "os0_payload": _freeze_os0(os0_for_orbit),
            "satellite": satellite,
            "est_satellite": est_satellite,
            "controller": controller,
            "estimator": estimator,
            "orbit_estimator": orbit_estimator,
        }

    runner = MonteCarloRunner(
        sim_func=_simulate_mc_worker,
        config_generator=_build_run_cfg,
        num_runs=int(num_runs),
        max_workers=max_workers,
    )

    raw = runner.run()

    ephem = Ephemeris()
    cleaned: List[Dict[str, Any]] = [r for r in raw if r is not None and isinstance(r, dict) and r.get("results") is not None]

    for item in cleaned:
        res = item["results"]
        res.os_hist = _thaw_os_hist(res.os_hist, ephem=ephem)
        res.est_os_hist = _thaw_os_hist(res.est_os_hist, ephem=ephem)

    runs = [entry["results"] for entry in cleaned]
    configs = [entry.get("applied", {}) for entry in cleaned]
    run_ids = [entry.get("run_id") for entry in cleaned]

    return MCSimulationResults(
        runs=runs,
        configs=configs,
        run_ids=run_ids,
    )

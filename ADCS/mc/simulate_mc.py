__all__ = ["simulate", "simulate_mc"]

import numpy as np
from typing import Optional, Any, Dict, Callable, List, Union
from copy import deepcopy

from ADCS.CONOPS.goals import Goal, GoalList
from ADCS.controller import Controller
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite

# Monte Carlo runner utilities
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)

# Import your MCConfig dataclass wherever you placed it
# from ADCS.helpers.mc.mc_config import MCConfig
# (If MCConfig lives in the same module, you don't need this import.)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ADCS.helpers.mc.mc_config import MCConfig  # pragma: no cover


def _is_callable_sampler(x: Any) -> bool:
    return callable(x)


def _sample_or_value(v: Any, rng: np.random.Generator) -> Any:
    """
    If v is callable, call it with rng; otherwise return v.
    This allows users to pass either concrete arrays OR samplers like lambda rng: ...
    """
    return v(rng) if _is_callable_sampler(v) else v


def _as_1d_float(arr: Any, expected_len: Optional[int] = None, name: str = "") -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if expected_len is not None and a.size != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {a.size}")
    return a


def _resolve_os0(base_os0: Orbital_State, override: Any) -> Orbital_State:
    """
    Orbit override rules:
      - None: keep base_os0
      - Orbital_State: use it directly
      - Orbit: derive os0 via orbit.get_os at base_os0.J2000 (keeps start time consistent)
    """
    if override is None:
        return base_os0

    if isinstance(override, Orbital_State):
        return override

    if isinstance(override, Orbit):
        return override.get_os(J2000=base_os0.J2000)

    raise TypeError("mc_config.os0/orbit override must be None, an Orbital_State, or an Orbit")


def _simulate_mc_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entry point for MonteCarloRunner. Must be top-level (picklable).
    Uses simulate() for the actual propagation.
    """
    slot_id = claim_worker_slot()
    run_id: int = int(cfg["run_id"])

    # We can't stream per-step progress from simulate() without modifying it,
    # so we do coarse updates: start and finished.
    try:
        update_worker_progress(slot_id, run_id, 0, 1)

        # Deterministic per-run RNG for any callable samplers used in MCConfig
        rng = np.random.default_rng(int(cfg["seed"]))

        # Unpack base inputs
        x_base: np.ndarray = np.asarray(cfg["x"], dtype=float).copy()
        satellite: Satellite = cfg["satellite"]
        est_satellite: Optional[EstimatedSatellite] = cfg.get("est_satellite", None)
        controller: Optional[Controller] = cfg.get("controller", None)
        estimator: Optional[Attitude_Estimator] = cfg.get("estimator", None)
        orbit_estimator: Optional[Orbit_Estimator] = cfg.get("orbit_estimator", None)
        goal_base: Optional[Union[Goal, GoalList]] = cfg.get("goal", None)
        os0_base: Orbital_State = cfg["os0"]
        dt_base: float = float(cfg["dt"])
        tf_base: float = float(cfg["tf"])

        mc_config = cfg.get("mc_config", None)

        # Apply MC overrides (data-only; no methods required)
        x0 = x_base.copy()
        goal = goal_base
        os0 = os0_base
        dt = dt_base
        tf = tf_base

        applied: Dict[str, Any] = {}

        if mc_config is not None:
            # dt / tf
            if getattr(mc_config, "dt", None) is not None:
                dt = float(mc_config.dt)
                applied["dt"] = dt
            if getattr(mc_config, "tf", None) is not None:
                tf = float(mc_config.tf)
                applied["tf"] = tf

            # w / q / h (allow arrays OR callables)
            if getattr(mc_config, "w", None) is not None:
                w = _sample_or_value(mc_config.w, rng)
                w = _as_1d_float(w, expected_len=3, name="mc_config.w")
                x0[:3] = w
                applied["w"] = w

            if getattr(mc_config, "q", None) is not None:
                q = _sample_or_value(mc_config.q, rng)
                q = _as_1d_float(q, expected_len=4, name="mc_config.q")
                x0[3:7] = q
                applied["q"] = q

            if getattr(mc_config, "h", None) is not None:
                h = _sample_or_value(mc_config.h, rng)
                h = _as_1d_float(h, expected_len=(len(x0) - 7), name="mc_config.h")
                x0[7:] = h
                applied["h"] = h

            # goal
            if getattr(mc_config, "goal", None) is not None:
                goal = _sample_or_value(mc_config.goal, rng)
                applied["goal"] = type(goal).__name__

            # orbit / os0 override
            # Support either mc_config.os0 (as in your example) OR mc_config.orbit
            orbit_override = None
            if hasattr(mc_config, "os0"):
                orbit_override = mc_config.os0
            if hasattr(mc_config, "orbit") and getattr(mc_config, "orbit") is not None:
                orbit_override = mc_config.orbit

            if orbit_override is not None:
                orbit_override = _sample_or_value(orbit_override, rng)
                os0 = _resolve_os0(os0_base, orbit_override)
                applied["os0"] = "Orbit" if isinstance(orbit_override, Orbit) else "Orbital_State"

        simulate_mod = __import__(__name__, fromlist=["tqdm"])
        _orig_tqdm = getattr(simulate_mod, "tqdm", None)
        setattr(simulate_mod, "tqdm", lambda it, **kwargs: it)

        try:
            sim_results = simulate(
                x=x0,
                satellite=satellite,
                est_satellite=est_satellite,
                controller=controller,
                estimator=estimator,
                orbit_estimator=orbit_estimator,
                goal=goal,
                os0=os0,
                dt=dt,
                tf=tf,
            )
        finally:
            # Restore tqdm
            if _orig_tqdm is not None:
                setattr(simulate_mod, "tqdm", _orig_tqdm)

        update_worker_progress(slot_id, run_id, 1, 1)

        return {
            "run_id": run_id,
            "seed": int(cfg["seed"]),
            "applied": applied,
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
) -> List[Dict[str, Any]]:
    if os0 is None:
        raise ValueError("os0 must be provided to simulate_mc().")
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length "
            f"{satellite.state_len}. It must be 7 + N_rw."
        )

    base_payload = dict(
        x=np.asarray(x, dtype=float).copy(),
        satellite=satellite,
        est_satellite=est_satellite,
        controller=controller,
        estimator=estimator,
        orbit_estimator=orbit_estimator,
        goal=goal,
        os0=os0,
        dt=float(dt),
        tf=float(tf),
        mc_config=mc_config, 
    )

    def _config_generator(run_id: int) -> Dict[str, Any]:
        cfg = dict(base_payload)
        cfg["run_id"] = int(run_id)
        cfg["seed"] = int(base_seed) + int(run_id)
        return cfg

    runner = MonteCarloRunner(
        sim_func=_simulate_mc_worker,
        config_generator=_config_generator,
        num_runs=int(num_runs),
        max_workers=max_workers if max_workers is not None else None,
    )
    return runner.run()

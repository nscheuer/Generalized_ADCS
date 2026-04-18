from __future__ import annotations

from typing import Optional

import numpy as np

from ADCS.CONOPS.goals import Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.controller import Controller
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.helpers.simresults import SimulationResults
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.remote.controller_rpc import (
    ComponentLocation,
    RemoteControllerProxy,
    RemoteAttitudeEstimatorProxy,
    RemoteOrbitEstimatorProxy,
    RemoteSimulationConfig,
)
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.simulate import simulate


def simulate_remote(
    x: np.ndarray,
    satellite: Satellite,
    os0: Orbital_State,
    *,
    controller: Optional[Controller] = None,
    estimator: Optional[Attitude_Estimator] = None,
    orbit_estimator: Optional[Orbit_Estimator] = None,
    goal: Optional[Goal | GoalList] = None,
    dt: float = 1.0,
    tf: float = 500.0,
    est_satellite: Optional[EstimatedSatellite] = None,
    remote: Optional[RemoteSimulationConfig] = None,
) -> SimulationResults:
    if remote is None:
        remote = RemoteSimulationConfig()

    controller_for_loop = controller
    if remote.controller == ComponentLocation.REMOTE:
        controller_for_loop = RemoteControllerProxy(
            host=remote.host,
            port=remote.port,
            timeout_s=remote.timeout_s,
            retries=remote.retries,
        )
        try:
            controller_for_loop.ping()
        except Exception as exc:
            raise ConnectionError(
                "Remote controller preflight failed. "
                f"Could not reach RPC server at {remote.host}:{remote.port}. "
                "Start debug/debug_remote/run_remote_universal.py on the remote host, "
                "or set ADCS_REMOTE_HOST/ADCS_REMOTE_PORT to a reachable endpoint."
            ) from exc

    estimator_for_loop = estimator
    if remote.estimator == ComponentLocation.REMOTE:
        estimator_for_loop = RemoteAttitudeEstimatorProxy(
            host=remote.host,
            port=remote.port,
            timeout_s=remote.timeout_s,
            retries=remote.retries,
        )
        try:
            estimator_for_loop.ping()
        except Exception as exc:
            raise ConnectionError(
                "Remote attitude-estimator preflight failed. "
                f"Could not reach RPC server at {remote.host}:{remote.port}. "
                "Start debug/debug_remote/run_remote_universal.py with an attitude-estimator component, "
                "or set ADCS_REMOTE_HOST/ADCS_REMOTE_PORT to a reachable endpoint."
            ) from exc

    orbit_estimator_for_loop = orbit_estimator
    if remote.orbit_estimator == ComponentLocation.REMOTE:
        orbit_estimator_for_loop = RemoteOrbitEstimatorProxy(
            host=remote.host,
            port=remote.port,
            timeout_s=remote.timeout_s,
            retries=remote.retries,
        )
        try:
            orbit_estimator_for_loop.ping()
        except Exception as exc:
            raise ConnectionError(
                "Remote orbit-estimator preflight failed. "
                f"Could not reach RPC server at {remote.host}:{remote.port}. "
                "Start debug/debug_remote/run_remote_universal.py with an orbit-estimator component, "
                "or set ADCS_REMOTE_HOST/ADCS_REMOTE_PORT to a reachable endpoint."
            ) from exc

    return simulate(
        x=x,
        satellite=satellite,
        est_satellite=est_satellite,
        controller=controller_for_loop,
        estimator=estimator_for_loop,
        orbit_estimator=orbit_estimator_for_loop,
        goal=goal,
        os0=os0,
        dt=dt,
        tf=tf,
    )
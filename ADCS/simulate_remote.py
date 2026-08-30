from __future__ import annotations

__all__ = ["simulate_remote"]

import time
from typing import Optional

import numpy as np

from ADCS.state import State

from ADCS.CONOPS.goals import Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.controller import Controller
from ADCS.estimators.old_attitude_estimators import Attitude_Estimator
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


def _as_float_array(values) -> np.ndarray:
    """Convert arbitrary timing history input into a finite 1-D float array.

    :param values:
        Input values from a run-history field. May be ``None`` or any array-like
        object.
    :type values:
        Any

    :return:
        Flattened array containing only finite float values.
    :rtype:
        numpy.ndarray
    """
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _fmt_seconds(value: float | None) -> str:
    """Format a duration in seconds and milliseconds for human-readable output.

    :param value:
        Duration in seconds.
    :type value:
        float or None

    :return:
        Formatted string, or ``"N/A"`` for missing/non-finite values.
    :rtype:
        str
    """
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.6f} s ({value * 1e3:.3f} ms)"


def _fmt_ms(value: float | None) -> str:
    """Format a duration in milliseconds for compact summary output.

    :param value:
        Duration in seconds.
    :type value:
        float or None

    :return:
        Millisecond-formatted string, or ``"N/A"`` for missing/non-finite values.
    :rtype:
        str
    """
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 1e3:.3f} ms"


def _safe_mean(arr: np.ndarray) -> float | None:
    """Return mean of an array or ``None`` when empty.

    :param arr:
        Input array.
    :type arr:
        numpy.ndarray

    :return:
        Arithmetic mean as float, or ``None`` if ``arr`` has no elements.
    :rtype:
        float or None
    """
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _safe_percentile(arr: np.ndarray, q: float) -> float | None:
    """Return percentile of an array or ``None`` when empty.

    :param arr:
        Input array.
    :type arr:
        numpy.ndarray

    :param q:
        Percentile in the range ``[0, 100]``.
    :type q:
        float

    :return:
        Requested percentile as float, or ``None`` if ``arr`` has no elements.
    :rtype:
        float or None
    """
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def _print_hil_timing_summary(
    *,
    results: SimulationResults,
    remote: RemoteSimulationConfig,
    wall_clock_s: float,
) -> None:
    """Print a concise timing breakdown for a remote simulation run.

    The summary reports wall-clock runtime and averaged/p-percentile timing for
    local environment work, dynamics propagation, RPC round-trip, and server
    compute time.

    :param results:
        Simulation results containing per-step timing histories.
    :type results:
        :class:`~ADCS.helpers.simresults.SimulationResults`

    :param remote:
        Remote simulation configuration used for endpoint context.
    :type remote:
        :class:`~ADCS.remote.controller_rpc.RemoteSimulationConfig`

    :param wall_clock_s:
        End-to-end wall-clock runtime in seconds.
    :type wall_clock_s:
        float
    """
    run = results.first()

    env_local = _as_float_array(getattr(run, "env_local_time_hist", None))
    dynamics = _as_float_array(getattr(run, "dynamics_time_hist", None))
    rpc_rtt = _as_float_array(getattr(run, "control_rpc_time_hist", None))
    rpc_server = _as_float_array(getattr(run, "control_rpc_server_time_hist", None))

    env_local_mean = _safe_mean(env_local)
    dynamics_mean = _safe_mean(dynamics)
    env_setup_mean = _safe_mean(env_local + dynamics) if (env_local.size and dynamics.size) else None

    rpc_rtt_mean = _safe_mean(rpc_rtt)
    rpc_server_mean = _safe_mean(rpc_server)
    rpc_comm_arr = np.asarray([], dtype=float)
    if rpc_rtt.size and rpc_server.size:
        n = min(rpc_rtt.size, rpc_server.size)
        rpc_comm_arr = np.maximum(rpc_rtt[:n] - rpc_server[:n], 0.0)
    elif rpc_rtt.size:
        rpc_comm_arr = rpc_rtt
    rpc_comm_mean = _safe_mean(rpc_comm_arr)

    step_count = int(len(getattr(run, "time_s", [])) if getattr(run, "time_s", None) is not None else 0)

    print("\n=== HIL Timing Summary (simulate_remote) ===")
    print(f"Endpoint: {remote.host}:{remote.port}")
    print(f"Steps: {step_count}")
    print(f"Total wall-clock run time: {_fmt_seconds(wall_clock_s)}")
    print(f"Average environmental setup (orbit calc/prop): {_fmt_seconds(env_setup_mean)}")
    print(f"Average communication uplink/downlink: {_fmt_ms(rpc_comm_mean)}")
    print(f"Average server calculation time: {_fmt_ms(rpc_server_mean)}")
    print(f"Average environment calculation time (locally): {_fmt_ms(env_local_mean)}")
    print(f"Average local dynamics propagation time: {_fmt_ms(dynamics_mean)}")
    print(f"RTT p50 / p95: {_fmt_ms(_safe_percentile(rpc_rtt, 50))} / {_fmt_ms(_safe_percentile(rpc_rtt, 95))}")
    print(
        f"Server p50 / p95: {_fmt_ms(_safe_percentile(rpc_server, 50))} / "
        f"{_fmt_ms(_safe_percentile(rpc_server, 95))}"
    )
    print("===========================================")


def simulate_remote(
    x: State,
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
    r"""
    Run an ADCS simulation with optional remoteized controller/estimator components.

    This wrapper configures RPC proxies based on ``remote`` placement settings,
    performs connectivity preflight checks, delegates execution to
    :func:`ADCS.simulate.simulate`, and prints a hardware-in-the-loop timing
    summary after completion.

    :param x:
        Initial true satellite state.
    :type x:
        ADCS.state.State

    :param satellite:
        True satellite model used for dynamics and sensor generation.
    :type satellite:
        :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`

    :param os0:
        Initial orbital state at simulation start.
    :type os0:
        :class:`~ADCS.orbits.orbital_state.Orbital_State`

    :param controller:
        Local controller implementation when controller placement is local.
    :type controller:
        :class:`~ADCS.controller.controller.Controller` or None

    :param estimator:
        Local attitude estimator when estimator placement is local.
    :type estimator:
        :class:`~ADCS.estimators.old_attitude_estimators.attitude_estimator.Attitude_Estimator` or None

    :param orbit_estimator:
        Local orbit estimator when orbit-estimator placement is local.
    :type orbit_estimator:
        :class:`~ADCS.estimators.orbit_estimators.orbit_estimator.Orbit_Estimator` or None

    :param goal:
        Optional pointing objective (single goal or goal list).
    :type goal:
        :class:`~ADCS.CONOPS.goals.goal.Goal`,
        :class:`~ADCS.CONOPS.goallist.goallist.GoalList`,
        or None

    :param dt:
        Simulation step size in seconds.
    :type dt:
        float

    :param tf:
        Simulation duration in seconds.
    :type tf:
        float

    :param est_satellite:
        Estimated satellite model for GNC components.
    :type est_satellite:
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite` or None

    :param remote:
        Remote execution placement and networking settings. If ``None``, defaults
        from :class:`~ADCS.remote.controller_rpc.RemoteSimulationConfig` are used.
    :type remote:
        :class:`~ADCS.remote.controller_rpc.RemoteSimulationConfig` or None

    :return:
        Full simulation history container.
    :rtype:
        :class:`~ADCS.helpers.simresults.SimulationResults`

    :raises ConnectionError:
        If a component is configured as remote and preflight RPC ping fails.
    """
    if remote is None:
        remote = RemoteSimulationConfig()

    wall_t0 = time.perf_counter()

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

    results = simulate(
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

    wall_clock_s = time.perf_counter() - wall_t0
    _print_hil_timing_summary(results=results, remote=remote, wall_clock_s=wall_clock_s)
    return results

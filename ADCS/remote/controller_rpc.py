from __future__ import annotations

__all__ = [
    "ComponentLocation",
    "RemoteSimulationConfig",
    "RemoteControllerProxy",
    "RemoteAttitudeEstimatorProxy",
    "RemoteOrbitEstimatorProxy",
    "serve_remote_components",
    "serve_remote_component",
    "serve_remote_controller",
]

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Sequence
from xmlrpc.client import ServerProxy, Transport
from xmlrpc.server import SimpleXMLRPCServer
import http.client
import time

import numpy as np

from ADCS.CONOPS.goals import ECI_Goal, Goal, No_Goal
from ADCS.controller.controller import Controller
from ADCS.estimators.old_attitude_estimators import Attitude_Estimator
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.orbits.orbital_state import Ephemeris, Orbital_State
from ADCS.state import EstimatorState, State


class ComponentLocation(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass
class RemoteSimulationConfig:
    controller: ComponentLocation = ComponentLocation.REMOTE
    estimator: ComponentLocation = ComponentLocation.LOCAL
    orbit_estimator: ComponentLocation = ComponentLocation.LOCAL
    host: str = "10.77.0.4"
    port: int = 5000
    timeout_s: float = 0.25
    retries: int = 0
    fallback: str = "raise"


class _TimeoutTransport(Transport):
    """XML-RPC transport with per-connection timeout enforcement."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self._timeout_s = float(timeout_s)

    def make_connection(self, host: str):
        return http.client.HTTPConnection(host, timeout=self._timeout_s)


def _goal_to_payload(goal: Goal | None) -> dict[str, Any]:
    """Serialize a goal object into an XML-RPC-safe dictionary.

    Only goal types currently used by the remote controller path are supported.
    The payload is intentionally explicit (with a ``kind`` tag) so the receiver
    can safely reconstruct the correct ADCS goal class.

    :param goal:
        Goal object to serialize. ``None`` is treated as :class:`No_Goal`.
    :type goal:
        :class:`~ADCS.CONOPS.goals.Goal` or None

    :return:
        XML-RPC-safe dictionary representation of the goal.
    :rtype:
        dict[str, Any]

    :raises NotImplementedError:
        If the goal type is not supported by the remote transport format.
    """

    if goal is None or isinstance(goal, No_Goal):
        return {"kind": "No_Goal"}

    if isinstance(goal, ECI_Goal):
        return {
            "kind": "ECI_Goal",
            "eci_vector": np.asarray(goal.eci_vector, dtype=float).reshape(3).tolist(),
            "boresight_name": goal.boresight_name,
        }

    raise NotImplementedError(
        f"Remote controller RPC only supports No_Goal and ECI_Goal for now, got {type(goal).__name__}."
    )


def _xmlrpc_safe(value: Any) -> Any:
    """Recursively convert ADCS/NumPy objects into XML-RPC-safe primitives.

    XML-RPC transports only basic Python scalar/list/dict types. This helper
    normalizes NumPy arrays/scalars and ADCS container objects so request and
    response payloads can be serialized by :mod:`xmlrpc.client`.

    :param value:
        Arbitrary value to normalize for XML-RPC transport.
    :type value:
        Any

    :return:
        A transport-safe representation of ``value``.
    :rtype:
        Any
    """

    if isinstance(value, EstimatedOrbital_State):
        return _estimated_orbital_state_to_payload(value)
    if isinstance(value, State):
        return _state_to_payload(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _xmlrpc_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_xmlrpc_safe(item) for item in value]
    return value


def _state_to_payload(state: State) -> dict[str, Any]:
    return _xmlrpc_safe({
        "schema_version": 1,
        "kind": "estimated" if isinstance(state, EstimatorState) else "physical",
        **state.to_dict(),
    })


def _state_from_payload(payload: dict[str, Any]) -> State:
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported state payload schema {payload.get('schema_version')!r}")
    if payload.get("kind") == "estimated":
        return EstimatorState.from_dict(payload)
    if payload.get("kind") == "physical":
        return State.from_dict(payload)
    raise ValueError(f"Unsupported state payload kind {payload.get('kind')!r}")


def _goal_from_payload(payload: dict[str, Any]) -> Goal:
    """Deserialize a goal payload back into an ADCS goal object.

    :param payload:
        Goal payload produced by :func:`_goal_to_payload`.
    :type payload:
        dict[str, Any]

    :return:
        Reconstructed ADCS goal instance.
    :rtype:
        :class:`~ADCS.CONOPS.goals.Goal`

    :raises NotImplementedError:
        If the payload encodes an unsupported goal ``kind``.
    """

    kind = payload.get("kind", "No_Goal")
    if kind == "No_Goal":
        return No_Goal()
    if kind == "ECI_Goal":
        return ECI_Goal(
            eci_vector=np.asarray(payload["eci_vector"], dtype=float).reshape(3),
            boresight_name=payload.get("boresight_name"),
        )
    raise NotImplementedError(f"Unsupported remote goal kind: {kind}")


def _print_remote_marker(marker: str) -> None:
    """Print a compact per-call marker in the server terminal.

    :param marker:
        Single-character marker identifying the RPC type (for example ``C``,
        ``A``, or ``O``).
    :type marker:
        str
    """

    print(marker, end="", flush=True)


def _os_to_payload(os0: Orbital_State | None) -> dict[str, Any] | None:
    """Serialize an orbital state for XML-RPC transport.

    :param os0:
        Orbital state to serialize.
    :type os0:
        :class:`~ADCS.orbits.orbital_state.Orbital_State` or None

    :return:
        ``None`` if input is ``None``; otherwise a dictionary payload containing
        orbital-state fields.
    :rtype:
        dict[str, Any] or None
    """

    return None if os0 is None else _xmlrpc_safe(os0.to_dict())


def _os_from_payload(payload: dict[str, Any] | None) -> Orbital_State | None:
    """Deserialize an orbital-state payload using a cached ephemeris.

    :param payload:
        Orbital-state dictionary payload, or ``None``.
    :type payload:
        dict[str, Any] or None

    :return:
        Reconstructed orbital state object, or ``None``.
    :rtype:
        :class:`~ADCS.orbits.orbital_state.Orbital_State` or None
    """

    if payload is None:
        return None

    return Orbital_State.from_dict(payload, ephem=_shared_ephemeris(), density_model=None, fast=True)


@lru_cache(maxsize=1)
def _shared_ephemeris() -> Ephemeris:
    """Return a process-wide cached ephemeris instance.

    Reusing the same ephemeris instance avoids repeated file loading and reduces
    per-request overhead during high-rate RPC deserialization.

    :return:
        Cached ephemeris object.
    :rtype:
        :class:`~ADCS.orbits.orbital_state.Ephemeris`
    """

    return Ephemeris()


def _estimated_orbital_state_to_payload(state: EstimatedOrbital_State | None) -> dict[str, Any] | None:
    """Serialize an estimated orbital state into an XML-RPC-safe dictionary.

    :param state:
        Estimated orbital state bundle to serialize.
    :type state:
        :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State` or None

    :return:
        Transport-safe dictionary containing ``os``, ``P``, and ``Q`` fields,
        or ``None`` if ``state`` is ``None``.
    :rtype:
        dict[str, Any] or None
    """

    if state is None:
        return None
    return {
        "os": _os_to_payload(state.os),
        "P": _xmlrpc_safe(np.asarray(state.P, dtype=float)),
        "Q": _xmlrpc_safe(np.asarray(state.Q, dtype=float)),
    }


def _estimated_orbital_state_from_payload(payload: dict[str, Any] | None) -> EstimatedOrbital_State | None:
    """Deserialize an estimated orbital-state payload.

    :param payload:
        Payload generated by :func:`_estimated_orbital_state_to_payload`.
    :type payload:
        dict[str, Any] or None

    :return:
        Reconstructed estimated orbital-state bundle.
    :rtype:
        :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State` or None
    """

    if payload is None:
        return None
    return EstimatedOrbital_State(
        os=_os_from_payload(payload["os"]),
        P=np.asarray(payload["P"], dtype=float),
        Q=np.asarray(payload["Q"], dtype=float),
    )


class RemoteControllerService:
    """Server-side RPC endpoint for controller `find_u` evaluation."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def ping(self) -> bool:
        """Return service liveness for preflight health checks.

        :return:
            ``True`` when the XML-RPC service is reachable.
        :rtype:
            bool
        """
        return True

    def find_u(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Evaluate controller command generation for one simulation step.

        :param payload:
            Request payload containing serialized controller inputs.
        :type payload:
            dict[str, Any]

        :return:
            Response payload with commanded control vector ``u`` and measured
            server compute duration in seconds.
        :rtype:
            dict[str, Any]
        """
        _print_remote_marker("C")
        start = time.perf_counter()
        x_hat = _state_from_payload(payload["x_hat"])
        sens = np.asarray(payload["sens"], dtype=float)
        os_hat = _os_from_payload(payload.get("os_hat"))
        goal = _goal_from_payload(payload.get("goal", {"kind": "No_Goal"}))

        u = self.controller.find_u(
            x_hat=x_hat,
            sens=sens,
            est_sat=self.controller.est_sat,
            os_hat=os_hat,
            goal=goal,
        )
        end = time.perf_counter()
        return {
            "u": np.asarray(u, dtype=float).reshape(-1).tolist(),
            "server_compute_s": end - start,
        }


class RemoteAttitudeEstimatorService:
    """Server-side RPC endpoint for attitude-estimator `update` calls."""

    def __init__(self, estimator: Attitude_Estimator) -> None:
        self.estimator = estimator

    def ping(self) -> bool:
        """Return service liveness for preflight health checks.

        :return:
            ``True`` when the XML-RPC service is reachable.
        :rtype:
            bool
        """
        return True

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one remote attitude-estimator update.

        :param payload:
            Request payload containing control input, sensor readings, and
            optional orbital-state estimate.
        :type payload:
            dict[str, Any]

        :return:
            Response payload containing the updated estimated state vector and
            server compute duration in seconds.
        :rtype:
            dict[str, Any]
        """
        _print_remote_marker("A")
        start = time.perf_counter()
        u = np.asarray(payload["u"], dtype=float)
        sensors = [np.asarray(sensor, dtype=float) for sensor in payload["sensors"]]
        os_hat = _os_from_payload(payload.get("os"))

        x_hat = self.estimator.update(u=u, sensors=sensors, os=os_hat)
        end = time.perf_counter()
        return {
            "x_hat": _state_to_payload(x_hat),
            "server_compute_s": end - start,
        }


class RemoteOrbitEstimatorService:
    """Server-side RPC endpoint for orbit-estimator `update` calls."""

    def __init__(self, estimator: Orbit_Estimator) -> None:
        self.estimator = estimator

    def ping(self) -> bool:
        """Return service liveness for preflight health checks.

        :return:
            ``True`` when the XML-RPC service is reachable.
        :rtype:
            bool
        """
        return True

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one remote orbit-estimator update.

        :param payload:
            Request payload containing GPS measurements and current J2000 time.
        :type payload:
            dict[str, Any]

        :return:
            Response payload containing serialized estimated orbital state and
            server compute duration in seconds.
        :rtype:
            dict[str, Any]
        """
        _print_remote_marker("O")
        start = time.perf_counter()
        gps_measurements = [np.asarray(measurement, dtype=float) for measurement in payload["GPS_measurements"]]
        J2000 = float(payload["J2000"])

        os_hat = self.estimator.update(GPS_measurements=gps_measurements, J2000=J2000)
        end = time.perf_counter()
        return {
            "os_hat": _estimated_orbital_state_to_payload(os_hat),
            "server_compute_s": end - start,
        }


class RemoteCompositeService:
    """Unified RPC service that can host controller and estimator components together."""

    def __init__(
        self,
        controller: Controller | None = None,
        estimator: Attitude_Estimator | None = None,
        orbit_estimator: Orbit_Estimator | None = None,
    ) -> None:
        self.controller = controller
        self.estimator = estimator
        self.orbit_estimator = orbit_estimator

    def ping(self) -> bool:
        """Return service liveness for preflight health checks.

        :return:
            ``True`` when the XML-RPC service is reachable.
        :rtype:
            bool
        """
        return True

    def find_u(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a remote controller ``find_u`` request.

        :param payload:
            Request payload containing controller inputs.
        :type payload:
            dict[str, Any]

        :return:
            Response payload containing control output ``u`` and server compute
            duration in seconds.
        :rtype:
            dict[str, Any]

        :raises RuntimeError:
            If no remote controller was configured on this server.
        """
        if self.controller is None:
            raise RuntimeError("No remote controller is configured on this server.")

        _print_remote_marker("C")
        start = time.perf_counter()
        x_hat = _state_from_payload(payload["x_hat"])
        sens = np.asarray(payload["sens"], dtype=float)
        os_hat = _os_from_payload(payload.get("os_hat"))
        goal = _goal_from_payload(payload.get("goal", {"kind": "No_Goal"}))

        u = self.controller.find_u(
            x_hat=x_hat,
            sens=sens,
            est_sat=self.controller.est_sat,
            os_hat=os_hat,
            goal=goal,
        )
        end = time.perf_counter()
        return {
            "u": np.asarray(u, dtype=float).reshape(-1).tolist(),
            "server_compute_s": end - start,
        }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a remote estimator update request.

        The request is routed to the orbit estimator or attitude estimator based
        on the explicit ``component`` field, or inferred payload keys for legacy
        callers.

        :param payload:
            Request payload identifying estimator target and inputs.
        :type payload:
            dict[str, Any]

        :return:
            Response payload containing estimator outputs and server compute time.
        :rtype:
            dict[str, Any]

        :raises RuntimeError:
            If the requested estimator component is not configured.
        :raises ValueError:
            If payload does not identify a supported estimator request type.
        """
        component = payload.get("component")

        if component == "orbit_estimator" or (component is None and "GPS_measurements" in payload):
            if self.orbit_estimator is None:
                raise RuntimeError("No remote orbit estimator is configured on this server.")
            _print_remote_marker("O")
            start = time.perf_counter()
            gps_measurements = [np.asarray(measurement, dtype=float) for measurement in payload["GPS_measurements"]]
            J2000 = float(payload["J2000"])

            os_hat = self.orbit_estimator.update(GPS_measurements=gps_measurements, J2000=J2000)
            end = time.perf_counter()
            return {
                "os_hat": _estimated_orbital_state_to_payload(os_hat),
                "server_compute_s": end - start,
            }

        if component == "attitude_estimator" or (component is None and "sensors" in payload):
            if self.estimator is None:
                raise RuntimeError("No remote attitude estimator is configured on this server.")
            _print_remote_marker("A")
            start = time.perf_counter()
            u = np.asarray(payload["u"], dtype=float)
            sensors = [np.asarray(sensor, dtype=float) for sensor in payload["sensors"]]
            os_hat = _os_from_payload(payload.get("os"))

            x_hat = self.estimator.update(u=u, sensors=sensors, os=os_hat)
            end = time.perf_counter()
            return {
                "x_hat": _state_to_payload(x_hat),
                "server_compute_s": end - start,
            }

        raise ValueError(
            "Remote update payload did not identify an orbit estimator or attitude estimator request."
        )


class _RemoteProxyBase:
    """Shared client-side RPC wrapper with retry and timing bookkeeping."""

    def __init__(self, *, host: str, port: int, timeout_s: float = 0.25, retries: int = 0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.last_roundtrip_s: float | None = None
        self.last_server_s: float | None = None
        self.roundtrip_hist: list[float] = []
        self.server_hist: list[float] = []
        self._proxy = ServerProxy(
            f"http://{self.host}:{self.port}",
            allow_none=True,
            transport=_TimeoutTransport(self.timeout_s),
        )

    def ping(self) -> bool:
        """Ping the remote XML-RPC endpoint.

        :return:
            ``True`` if the remote endpoint responds to health checks.
        :rtype:
            bool
        """
        return bool(self._proxy.ping())

    def _call(self, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform an RPC call with retry and timing bookkeeping.

        :param method_name:
            Remote method name to invoke on the XML-RPC server.
        :type method_name:
            str

        :param payload:
            XML-RPC-safe request payload.
        :type payload:
            dict[str, Any]

        :return:
            RPC response dictionary.
        :rtype:
            dict[str, Any]

        :raises RuntimeError:
            If all retries fail and no response is received.
        """
        start = time.perf_counter()
        response = None
        last_error: Exception | None = None
        remote_method = getattr(self._proxy, method_name)
        for _ in range(self.retries + 1):
            try:
                response = remote_method(payload)
                break
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
        if response is None:
            raise RuntimeError(f"Remote call {method_name} failed for {self.host}:{self.port}") from last_error

        end = time.perf_counter()
        self.last_roundtrip_s = end - start
        self.last_server_s = float(response.get("server_compute_s", float("nan")))
        self.roundtrip_hist.append(self.last_roundtrip_s)
        self.server_hist.append(self.last_server_s)
        return response


class RemoteControllerProxy:
    """Client-side proxy exposing controller-compatible `find_u` calls."""

    def __init__(self, *, host: str, port: int, timeout_s: float = 0.25, retries: int = 0) -> None:
        self._base = _RemoteProxyBase(host=host, port=port, timeout_s=timeout_s, retries=retries)

    @property
    def host(self) -> str:
        return self._base.host

    @property
    def port(self) -> int:
        return self._base.port

    @property
    def last_roundtrip_s(self) -> float | None:
        return self._base.last_roundtrip_s

    @property
    def last_server_s(self) -> float | None:
        return self._base.last_server_s

    @property
    def roundtrip_hist(self) -> list[float]:
        return self._base.roundtrip_hist

    @property
    def server_hist(self) -> list[float]:
        return self._base.server_hist

    def ping(self) -> bool:
        """Ping the remote server used by this controller proxy.

        :return:
            ``True`` if the remote endpoint responds.
        :rtype:
            bool
        """
        return self._base.ping()

    def find_u(self, x_hat: State | EstimatorState, sens: np.ndarray, est_sat: Any, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        """Execute remote controller command synthesis.

        :param x_hat:
            State estimate for controller input.
        :type x_hat:
            ADCS.state.State or ADCS.state.EstimatorState

        :param sens:
            Sensor measurement vector for controller input.
        :type sens:
            numpy.ndarray

        :param est_sat:
            Estimated satellite model (unused for transport parity).
        :type est_sat:
            Any

        :param os_hat:
            Orbital state estimate.
        :type os_hat:
            :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param goal:
            Active guidance goal for the current step.
        :type goal:
            :class:`~ADCS.CONOPS.goals.Goal` or None

        :return:
            Control command vector from the remote controller.
        :rtype:
            numpy.ndarray
        """
        payload = {
            "x_hat": _state_to_payload(x_hat),
            "sens": np.asarray(sens, dtype=float).reshape(-1).tolist(),
            "os_hat": _os_to_payload(os_hat),
            "goal": _goal_to_payload(goal),
        }
        response = self._base._call("find_u", payload)
        return np.asarray(response["u"], dtype=float).reshape(-1)


class RemoteAttitudeEstimatorProxy:
    """Client-side proxy exposing attitude-estimator-compatible `update` calls."""

    def __init__(self, *, host: str, port: int, timeout_s: float = 0.25, retries: int = 0) -> None:
        self._base = _RemoteProxyBase(host=host, port=port, timeout_s=timeout_s, retries=retries)

    @property
    def host(self) -> str:
        return self._base.host

    @property
    def port(self) -> int:
        return self._base.port

    @property
    def last_roundtrip_s(self) -> float | None:
        return self._base.last_roundtrip_s

    @property
    def last_server_s(self) -> float | None:
        return self._base.last_server_s

    @property
    def roundtrip_hist(self) -> list[float]:
        return self._base.roundtrip_hist

    @property
    def server_hist(self) -> list[float]:
        return self._base.server_hist

    def ping(self) -> bool:
        """Ping the remote server used by this attitude-estimator proxy.

        :return:
            ``True`` if the remote endpoint responds.
        :rtype:
            bool
        """
        return self._base.ping()

    def update(self, u: np.ndarray, sensors: Sequence[np.ndarray], os: Orbital_State) -> EstimatorState:
        """Execute one remote attitude-estimator update.

        :param u:
            Applied control command vector.
        :type u:
            numpy.ndarray

        :param sensors:
            Sensor measurement list for the estimator update.
        :type sensors:
            list[numpy.ndarray]

        :param os:
            Orbital-state estimate used by the estimator.
        :type os:
            :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return:
            Updated attitude estimator state.
        :rtype:
            :class:`~ADCS.state.EstimatorState`
        """
        payload = {
            "component": "attitude_estimator",
            "u": np.asarray(u, dtype=float).reshape(-1).tolist(),
            "sensors": [_xmlrpc_safe(np.asarray(sensor, dtype=float)) for sensor in sensors],
            "os": _os_to_payload(os),
        }
        response = self._base._call("update", payload)
        return _state_from_payload(response["x_hat"])


class RemoteOrbitEstimatorProxy:
    """Client-side proxy exposing orbit-estimator-compatible `update` calls."""

    def __init__(self, *, host: str, port: int, timeout_s: float = 0.25, retries: int = 0) -> None:
        self._base = _RemoteProxyBase(host=host, port=port, timeout_s=timeout_s, retries=retries)

    @property
    def host(self) -> str:
        return self._base.host

    @property
    def port(self) -> int:
        return self._base.port

    @property
    def last_roundtrip_s(self) -> float | None:
        return self._base.last_roundtrip_s

    @property
    def last_server_s(self) -> float | None:
        return self._base.last_server_s

    @property
    def roundtrip_hist(self) -> list[float]:
        return self._base.roundtrip_hist

    @property
    def server_hist(self) -> list[float]:
        return self._base.server_hist

    def ping(self) -> bool:
        """Ping the remote server used by this orbit-estimator proxy.

        :return:
            ``True`` if the remote endpoint responds.
        :rtype:
            bool
        """
        return self._base.ping()

    def update(self, GPS_measurements: Sequence[np.ndarray], J2000: float) -> EstimatedOrbital_State:
        """Execute one remote orbit-estimator update.

        :param GPS_measurements:
            List of GPS measurement vectors.
        :type GPS_measurements:
            list[numpy.ndarray]

        :param J2000:
            Current simulation time in Julian centuries from J2000.
        :type J2000:
            float

        :return:
            Updated estimated orbital state.
        :rtype:
            :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State`
        """
        payload = {
            "component": "orbit_estimator",
            "GPS_measurements": [_xmlrpc_safe(np.asarray(measurement, dtype=float)) for measurement in GPS_measurements],
            "J2000": float(J2000),
        }
        response = self._base._call("update", payload)
        return _estimated_orbital_state_from_payload(response["os_hat"])


def serve_remote_components(
    *,
    controller: Controller | None = None,
    estimator: Attitude_Estimator | None = None,
    orbit_estimator: Orbit_Estimator | None = None,
    host: str = "0.0.0.0",
    port: int = 5000,
) -> None:
    """Serve any combination of ADCS components over a single XML-RPC endpoint.

    :param controller:
        Optional controller component handling ``find_u`` requests.
    :type controller:
        :class:`~ADCS.controller.controller.Controller` or None

    :param estimator:
        Optional attitude estimator handling attitude ``update`` requests.
    :type estimator:
        :class:`~ADCS.estimators.old_attitude_estimators.Attitude_Estimator` or None

    :param orbit_estimator:
        Optional orbit estimator handling orbit ``update`` requests.
    :type orbit_estimator:
        :class:`~ADCS.estimators.orbit_estimators.Orbit_Estimator` or None

    :param host:
        Bind address for the server.
    :type host:
        str

    :param port:
        Bind port for the server.
    :type port:
        int

    :raises ValueError:
        If no components are provided.
    """

    if controller is None and estimator is None and orbit_estimator is None:
        raise ValueError("At least one remote component must be provided.")

    service = RemoteCompositeService(
        controller=controller,
        estimator=estimator,
        orbit_estimator=orbit_estimator,
    )
    server = SimpleXMLRPCServer((host, int(port)), allow_none=True, logRequests=False)
    server.register_introspection_functions()
    server.register_instance(service, allow_dotted_names=False)
    print(
        "[RemoteCompositeService] listening on "
        f"{host}:{port} "
        f"(controller={'yes' if controller is not None else 'no'}, "
        f"attitude_estimator={'yes' if estimator is not None else 'no'}, "
        f"orbit_estimator={'yes' if orbit_estimator is not None else 'no'})"
    )
    server.serve_forever()


def serve_remote_component(component: Any, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Serve one ADCS component by dispatching to the composite helper.

    :param component:
        Component instance to serve.
    :type component:
        Any

    :param host:
        Bind address for the server.
    :type host:
        str

    :param port:
        Bind port for the server.
    :type port:
        int

    :raises TypeError:
        If ``component`` is not one of the supported ADCS component classes.
    """

    if isinstance(component, Controller):
        serve_remote_components(controller=component, host=host, port=port)
        return
    elif isinstance(component, Attitude_Estimator):
        serve_remote_components(estimator=component, host=host, port=port)
        return
    elif isinstance(component, Orbit_Estimator):
        serve_remote_components(orbit_estimator=component, host=host, port=port)
        return
    else:
        raise TypeError(
            "Unsupported remote component type. Expected a Controller, Attitude_Estimator, or Orbit_Estimator."
        )


def serve_remote_controller(controller: Any, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Backward-compatible controller-only server entry point.

    :param controller:
        Controller instance to serve.
    :type controller:
        Any

    :param host:
        Bind address for the server.
    :type host:
        str

    :param port:
        Bind port for the server.
    :type port:
        int
    """

    serve_remote_component(controller, host=host, port=port)

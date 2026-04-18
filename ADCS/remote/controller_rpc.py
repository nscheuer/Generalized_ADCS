from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any
from xmlrpc.client import ServerProxy, Transport
from xmlrpc.server import SimpleXMLRPCServer
import http.client
import time

import numpy as np

from ADCS.CONOPS.goals import ECI_Goal, Goal, No_Goal
from ADCS.controller.controller import Controller
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.orbits.orbital_state import Ephemeris, Orbital_State


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
    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self._timeout_s = float(timeout_s)

    def make_connection(self, host: str):
        return http.client.HTTPConnection(host, timeout=self._timeout_s)


def _goal_to_payload(goal: Goal | None) -> dict[str, Any]:
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
    if isinstance(value, EstimatedOrbital_State):
        return _estimated_orbital_state_to_payload(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _xmlrpc_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_xmlrpc_safe(item) for item in value]
    return value


def _goal_from_payload(payload: dict[str, Any]) -> Goal:
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
    print(marker, end="", flush=True)


def _os_to_payload(os0: Orbital_State | None) -> dict[str, Any] | None:
    return None if os0 is None else _xmlrpc_safe(os0.to_dict())


def _os_from_payload(payload: dict[str, Any] | None) -> Orbital_State | None:
    if payload is None:
        return None

    return Orbital_State.from_dict(payload, ephem=_shared_ephemeris(), density_model=None, fast=True)


@lru_cache(maxsize=1)
def _shared_ephemeris() -> Ephemeris:
    return Ephemeris()


def _estimated_orbital_state_to_payload(state: EstimatedOrbital_State | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "os": _os_to_payload(state.os),
        "P": _xmlrpc_safe(np.asarray(state.P, dtype=float)),
        "Q": _xmlrpc_safe(np.asarray(state.Q, dtype=float)),
    }


def _estimated_orbital_state_from_payload(payload: dict[str, Any] | None) -> EstimatedOrbital_State | None:
    if payload is None:
        return None
    return EstimatedOrbital_State(
        os=_os_from_payload(payload["os"]),
        P=np.asarray(payload["P"], dtype=float),
        Q=np.asarray(payload["Q"], dtype=float),
    )


class RemoteControllerService:
    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def ping(self) -> bool:
        return True

    def find_u(self, payload: dict[str, Any]) -> dict[str, Any]:
        _print_remote_marker("C: Controller called ")
        start = time.perf_counter()
        x_hat = np.asarray(payload["x_hat"], dtype=float)
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
    def __init__(self, estimator: Attitude_Estimator) -> None:
        self.estimator = estimator

    def ping(self) -> bool:
        return True

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        _print_remote_marker("A: Attitude Estimator called ")
        start = time.perf_counter()
        u = np.asarray(payload["u"], dtype=float)
        sensors = [np.asarray(sensor, dtype=float) for sensor in payload["sensors"]]
        os_hat = _os_from_payload(payload.get("os"))

        x_hat = self.estimator.update(u=u, sensors=sensors, os=os_hat)
        end = time.perf_counter()
        return {
            "x_hat": np.asarray(x_hat, dtype=float).reshape(-1).tolist(),
            "server_compute_s": end - start,
        }


class RemoteOrbitEstimatorService:
    def __init__(self, estimator: Orbit_Estimator) -> None:
        self.estimator = estimator

    def ping(self) -> bool:
        return True

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        _print_remote_marker("O: Orbit Estimator called ")
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
        return True

    def find_u(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError("No remote controller is configured on this server.")

        _print_remote_marker("C: Controller called ")
        start = time.perf_counter()
        x_hat = np.asarray(payload["x_hat"], dtype=float)
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
        component = payload.get("component")

        if component == "orbit_estimator" or (component is None and "GPS_measurements" in payload):
            if self.orbit_estimator is None:
                raise RuntimeError("No remote orbit estimator is configured on this server.")
            _print_remote_marker("O: Orbit Estimator called ")
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
            _print_remote_marker("A: Attitude Estimator called ")
            start = time.perf_counter()
            u = np.asarray(payload["u"], dtype=float)
            sensors = [np.asarray(sensor, dtype=float) for sensor in payload["sensors"]]
            os_hat = _os_from_payload(payload.get("os"))

            x_hat = self.estimator.update(u=u, sensors=sensors, os=os_hat)
            end = time.perf_counter()
            return {
                "x_hat": np.asarray(x_hat, dtype=float).reshape(-1).tolist(),
                "server_compute_s": end - start,
            }

        raise ValueError(
            "Remote update payload did not identify an orbit estimator or attitude estimator request."
        )


class _RemoteProxyBase:
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
        return bool(self._proxy.ping())

    def _call(self, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        return self._base.ping()

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: Any, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        payload = {
            "x_hat": np.asarray(x_hat, dtype=float).reshape(-1).tolist(),
            "sens": np.asarray(sens, dtype=float).reshape(-1).tolist(),
            "os_hat": _os_to_payload(os_hat),
            "goal": _goal_to_payload(goal),
        }
        response = self._base._call("find_u", payload)
        return np.asarray(response["u"], dtype=float).reshape(-1)


class RemoteAttitudeEstimatorProxy:
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
        return self._base.ping()

    def update(self, u: np.ndarray, sensors: list[np.ndarray], os: Orbital_State) -> np.ndarray:
        payload = {
            "component": "attitude_estimator",
            "u": np.asarray(u, dtype=float).reshape(-1).tolist(),
            "sensors": [_xmlrpc_safe(np.asarray(sensor, dtype=float)) for sensor in sensors],
            "os": _os_to_payload(os),
        }
        response = self._base._call("update", payload)
        return np.asarray(response["x_hat"], dtype=float).reshape(-1)


class RemoteOrbitEstimatorProxy:
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
        return self._base.ping()

    def update(self, GPS_measurements: list[np.ndarray], J2000: float) -> EstimatedOrbital_State:
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
    serve_remote_component(controller, host=host, port=port)
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from xmlrpc.client import ServerProxy, Transport
from xmlrpc.server import SimpleXMLRPCServer
import http.client
import time

import numpy as np

from ADCS.CONOPS.goals import ECI_Goal, Goal, No_Goal
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


def _os_to_payload(os0: Orbital_State | None) -> dict[str, Any] | None:
    return None if os0 is None else _xmlrpc_safe(os0.to_dict())


def _os_from_payload(payload: dict[str, Any] | None) -> Orbital_State | None:
    if payload is None:
        return None
    return Orbital_State.from_dict(payload, ephem=Ephemeris(), density_model=None, fast=True)


class RemoteControllerService:
    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def ping(self) -> bool:
        return True

    def find_u(self, payload: dict[str, Any]) -> dict[str, Any]:
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


class RemoteControllerProxy:
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

    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: Any, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        payload = {
            "x_hat": np.asarray(x_hat, dtype=float).reshape(-1).tolist(),
            "sens": np.asarray(sens, dtype=float).reshape(-1).tolist(),
            "os_hat": _os_to_payload(os_hat),
            "goal": _goal_to_payload(goal),
        }

        start = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                response = self._proxy.find_u(payload)
                break
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
        if response is None:
            raise RuntimeError(f"Remote controller call failed for {self.host}:{self.port}") from last_error

        end = time.perf_counter()
        self.last_roundtrip_s = end - start
        self.last_server_s = float(response.get("server_compute_s", float("nan")))
        self.roundtrip_hist.append(self.last_roundtrip_s)
        self.server_hist.append(self.last_server_s)
        return np.asarray(response["u"], dtype=float).reshape(-1)


def serve_remote_controller(controller: Any, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    service = RemoteControllerService(controller)
    server = SimpleXMLRPCServer((host, int(port)), allow_none=True, logRequests=False)
    server.register_introspection_functions()
    server.register_instance(service, allow_dotted_names=False)
    print(f"[RemoteControllerService] listening on {host}:{port}")
    server.serve_forever()
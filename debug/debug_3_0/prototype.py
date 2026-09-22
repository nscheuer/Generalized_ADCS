"""A small block-graph satellite simulator used to explore a 3.0 architecture.

The physics are deliberately compact, but every requested subsystem is real:

* J2 or J2/J3/J4 orbit propagation with RK2/RK4;
* attitude-dependent aerodynamic force and torque;
* gyro, magnetometer and sun-vector measurements with noise;
* a small multiplicative-EKF-inspired attitude estimator;
* three magnetorquers and three reaction wheels with held actuator noise;
* gravity-gradient and aerodynamic disturbance torques;
* Numba-compiled RK2/RK4 attitude propagation;
* fixed minimum-tick and asynchronous event execution engines.

All blocks communicate through timestamped signals.  This is the important part
of the prototype: numerical models do not know which scheduler executes them.
"""

from __future__ import annotations

import copy
import heapq
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
from numba import njit


MU_EARTH = 3.986004418e14
R_EARTH = 6_378_137.0
J2 = 1.08262668e-3
J3 = -2.5324105e-6
J4 = -1.6198976e-6
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])
SUN_ECI = np.array([0.92541658, 0.33682409, 0.17364818])
INERTIA = np.diag(np.array([0.035, 0.045, 0.052]))
INV_INERTIA = np.diag(1.0 / np.diag(INERTIA))
MASS = 12.0


def _unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return np.asarray(x, dtype=float) / max(n, 1e-15)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first."""
    return np.array(
        [
            a[0] * b[0] - np.dot(a[1:], b[1:]),
            a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
            a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.r_[q[0], -q[1:]]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body-to-inertial rotation matrix for a scalar-first quaternion."""
    q = _unit(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotvec_quat(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return _unit(np.r_[1.0, 0.5 * v])
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * v / angle]


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@njit
def _attitude_rhs(y: np.ndarray, torque: np.ndarray, rw_torque: np.ndarray) -> np.ndarray:
    w = y[0:3]
    q = y[3:7]
    h = y[7:10]
    angular_momentum = INERTIA @ w + h
    wdot = INV_INERTIA @ (torque - np.cross(w, angular_momentum))
    qdot = np.empty(4)
    qdot[0] = -0.5 * (q[1] * w[0] + q[2] * w[1] + q[3] * w[2])
    qdot[1] = 0.5 * (q[0] * w[0] + q[2] * w[2] - q[3] * w[1])
    qdot[2] = 0.5 * (q[0] * w[1] + q[3] * w[0] - q[1] * w[2])
    qdot[3] = 0.5 * (q[0] * w[2] + q[1] * w[1] - q[2] * w[0])
    out = np.empty(10)
    out[0:3] = wdot
    out[3:7] = qdot
    out[7:10] = -rw_torque
    return out


@njit
def _attitude_integrate(y: np.ndarray, dt: float, torque: np.ndarray, rw_torque: np.ndarray, order: int) -> np.ndarray:
    if dt <= 0.0:
        return y.copy()
    if order == 2:
        k1 = _attitude_rhs(y, torque, rw_torque)
        k2 = _attitude_rhs(y + 0.5 * dt * k1, torque, rw_torque)
        out = y + dt * k2
    else:
        k1 = _attitude_rhs(y, torque, rw_torque)
        k2 = _attitude_rhs(y + 0.5 * dt * k1, torque, rw_torque)
        k3 = _attitude_rhs(y + 0.5 * dt * k2, torque, rw_torque)
        k4 = _attitude_rhs(y + dt * k3, torque, rw_torque)
        out = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    qn = math.sqrt(np.dot(out[3:7], out[3:7]))
    out[3:7] /= qn
    return out


def zonal_acceleration(r: np.ndarray, degree: int) -> np.ndarray:
    """Earth acceleration from a compact zonal-potential gradient."""
    radius = float(np.linalg.norm(r))
    rhat = r / radius
    s = float(rhat[2])
    radial = -MU_EARTH / radius**2
    ds = 0.0
    terms: list[tuple[int, float, float, float]] = []
    if degree >= 2:
        terms.append((2, J2, 0.5 * (3 * s * s - 1), 3 * s))
    if degree >= 3:
        terms.append((3, J3, 0.5 * (5 * s**3 - 3 * s), 0.5 * (15 * s * s - 3)))
    if degree >= 4:
        terms.append((4, J4, (35 * s**4 - 30 * s * s + 3) / 8, (140 * s**3 - 60 * s) / 8))
    for n, jn, pn, dpn in terms:
        scale = jn * R_EARTH**n / radius ** (n + 2)
        radial += MU_EARTH * (n + 1) * scale * pn
        ds -= MU_EARTH * jn * R_EARTH**n / radius ** (n + 1) * dpn
    return radial * rhat + (ds / radius) * (np.array([0.0, 0.0, 1.0]) - s * rhat)


def orbit_rhs(y: np.ndarray, force_eci: np.ndarray, degree: int) -> np.ndarray:
    return np.r_[y[3:6], zonal_acceleration(y[0:3], degree) + force_eci / MASS]


def orbit_integrate(y: np.ndarray, dt: float, force_eci: np.ndarray, degree: int, scheme: str) -> np.ndarray:
    if dt <= 0:
        return y.copy()
    k1 = orbit_rhs(y, force_eci, degree)
    if scheme == "rk2":
        return y + dt * orbit_rhs(y + 0.5 * dt * k1, force_eci, degree)
    k2 = orbit_rhs(y + 0.5 * dt * k1, force_eci, degree)
    k3 = orbit_rhs(y + 0.5 * dt * k2, force_eci, degree)
    k4 = orbit_rhs(y + dt * k3, force_eci, degree)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def magnetic_field_eci(r: np.ndarray) -> np.ndarray:
    """Simple aligned-dipole magnetic field, in tesla."""
    rhat = _unit(r)
    dipole = np.array([0.0, 0.0, 1.0])
    return 3.12e-5 * (R_EARTH / np.linalg.norm(r)) ** 3 * (3 * rhat * np.dot(dipole, rhat) - dipole)


@dataclass(frozen=True)
class Timing:
    period: float
    compute_time: float = 0.0
    output_delay: float = 0.0
    phase: float = 0.0

    @property
    def latency(self) -> float:
        return self.compute_time + self.output_delay


def default_timings(*, asynchronous: bool = False) -> dict[str, Timing]:
    delays = asynchronous
    return {
        "orbit": Timing(0.20, 0.004 if delays else 0.0, 0.0),
        "attitude": Timing(0.02, 0.001 if delays else 0.0, 0.0),
        "disturbances": Timing(0.10, 0.006 if delays else 0.0, 0.004 if delays else 0.0),
        "sensors": Timing(0.10, 0.008 if delays else 0.0, 0.027 if delays else 0.0),
        "estimator": Timing(0.20, 0.045 if delays else 0.0, 0.035 if delays else 0.0, 0.01 if delays else 0.0),
        "controller": Timing(0.20, 0.018 if delays else 0.0, 0.022 if delays else 0.0, 0.02),
        "actuators": Timing(0.04, 0.004 if delays else 0.0, 0.011 if delays else 0.0),
        "logger": Timing(0.02),
    }


@dataclass(frozen=True)
class ModelConfig:
    duration: float = 20.0
    seed: int = 7
    orbit_degree: int = 4
    orbit_integrator: str = "rk4"
    attitude_integrator: str = "rk4"
    timings: dict[str, Timing] = field(default_factory=default_timings)
    gyro_noise_std: float = 2.0e-4
    magnetometer_noise_std: float = 2.0e-7
    sun_noise_std: float = 2.0e-3
    rw_noise_std: float = 2.0e-5
    mtq_noise_std: float = 2.0e-3

    def __post_init__(self) -> None:
        if self.orbit_degree not in (2, 4):
            raise ValueError("orbit_degree must be 2 (J2) or 4 (J2/J3/J4)")
        if self.orbit_integrator not in ("rk2", "rk4"):
            raise ValueError("orbit_integrator must be 'rk2' or 'rk4'")
        if self.attitude_integrator not in ("rk2", "rk4"):
            raise ValueError("attitude_integrator must be 'rk2' or 'rk4'")


@dataclass
class Signal:
    value: Any
    sampled_at: float
    delivered_at: float
    source: str


@dataclass
class SensorPacket:
    gyro: np.ndarray
    magnetic_body: np.ndarray
    sun_body: np.ndarray
    sampled_at: float


@dataclass
class EstimatePacket:
    q: np.ndarray
    w: np.ndarray
    gyro_bias: np.ndarray
    covariance: np.ndarray
    measurement_time: float


@dataclass
class ControlPacket:
    mtq_dipole: np.ndarray
    rw_torque: np.ndarray
    computed_from: float


@dataclass
class ActuatorPacket:
    body_torque: np.ndarray
    rw_torque: np.ndarray
    command_time: float


@dataclass
class DisturbancePacket:
    force_eci: np.ndarray
    torque_body: np.ndarray


@dataclass
class TraceEntry:
    block: str
    start: float
    finish: float
    delivery: float


class Context:
    def __init__(self) -> None:
        self.signals: dict[str, Signal] = {}
        self.trace: list[TraceEntry] = []
        self.history: dict[str, list[Any]] = {}

    def put(self, name: str, value: Any, sampled_at: float, delivered_at: float, source: str) -> None:
        self.signals[name] = Signal(copy.deepcopy(value), sampled_at, delivered_at, source)

    def get(self, name: str) -> Any:
        return self.signals[name].value

    def signal(self, name: str) -> Signal:
        return self.signals[name]

    def append(self, name: str, value: Any) -> None:
        self.history.setdefault(name, []).append(copy.deepcopy(value))


class Block:
    name = "block"
    priority = 50
    inputs: dict[str, type] = {}
    outputs: dict[str, type] = {}

    def __init__(self, timing: Timing) -> None:
        self.timing = timing

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        raise NotImplementedError


class OrbitBlock(Block):
    name = "orbit"
    priority = 10
    inputs = {"disturbance": DisturbancePacket}
    outputs = {"orbit": np.ndarray}

    def __init__(self, timing: Timing, config: ModelConfig, initial: np.ndarray) -> None:
        super().__init__(timing)
        self.config = config
        self.state = initial.copy()
        self.last_time = 0.0

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        force = ctx.get("disturbance").force_eci
        self.state = orbit_integrate(
            self.state, t - self.last_time, force, self.config.orbit_degree, self.config.orbit_integrator
        )
        self.last_time = t
        return {"orbit": self.state}


class AttitudeBlock(Block):
    name = "attitude"
    priority = 20
    inputs = {"actuator": ActuatorPacket, "disturbance": DisturbancePacket}
    outputs = {"attitude": np.ndarray}

    def __init__(self, timing: Timing, config: ModelConfig, initial: np.ndarray) -> None:
        super().__init__(timing)
        self.order = 2 if config.attitude_integrator == "rk2" else 4
        self.state = initial.copy()
        self.last_time = 0.0

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        actuator = ctx.get("actuator")
        disturbance = ctx.get("disturbance")
        torque = actuator.body_torque + disturbance.torque_body
        self.state = _attitude_integrate(self.state, t - self.last_time, torque, actuator.rw_torque, self.order)
        self.last_time = t
        return {"attitude": self.state}


class DisturbanceBlock(Block):
    name = "disturbances"
    priority = 30
    inputs = {"orbit": np.ndarray, "attitude": np.ndarray}
    outputs = {"disturbance": DisturbancePacket}

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        orbit = ctx.get("orbit")
        attitude = ctx.get("attitude")
        r, v = orbit[:3], orbit[3:]
        q = attitude[3:7]
        body_to_eci = quat_to_matrix(q)
        relative_v = v - np.cross(OMEGA_EARTH, r)
        flow_body = body_to_eci.T @ relative_v
        flow_hat = _unit(flow_body)
        face_areas = np.array([0.10 * 0.20, 0.10 * 0.30, 0.20 * 0.30])
        projected_area = float(np.dot(face_areas, np.abs(flow_hat)))
        altitude = np.linalg.norm(r) - R_EARTH
        density = 6.0e-13 * math.exp(-(altitude - 500_000.0) / 60_000.0)
        drag_force = -0.5 * density * 2.2 * projected_area * np.linalg.norm(relative_v) * relative_v
        drag_force_body = body_to_eci.T @ drag_force
        cp_offset = np.array([0.012, -0.007, 0.018])
        drag_torque = np.cross(cp_offset, drag_force_body)
        rhat_body = body_to_eci.T @ _unit(r)
        gg_torque = 3 * MU_EARTH / np.linalg.norm(r) ** 3 * np.cross(rhat_body, INERTIA @ rhat_body)
        return {"disturbance": DisturbancePacket(drag_force, drag_torque + gg_torque)}


class SensorBlock(Block):
    name = "sensors"
    priority = 40
    inputs = {"orbit": np.ndarray, "attitude": np.ndarray}
    outputs = {"sensors": SensorPacket}

    def __init__(self, timing: Timing, config: ModelConfig) -> None:
        super().__init__(timing)
        self.config = config
        self.gyro_bias = np.array([2.0e-4, -1.0e-4, 1.5e-4])

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        orbit = ctx.get("orbit")
        attitude = ctx.get("attitude")
        w, q = attitude[:3], attitude[3:7]
        inertial_to_body = quat_to_matrix(q).T
        b_body = inertial_to_body @ magnetic_field_eci(orbit[:3])
        sun_body = inertial_to_body @ SUN_ECI
        self.gyro_bias += rng.normal(0.0, 3.0e-7, 3)
        packet = SensorPacket(
            gyro=w + self.gyro_bias + rng.normal(0.0, self.config.gyro_noise_std, 3),
            magnetic_body=b_body + rng.normal(0.0, self.config.magnetometer_noise_std, 3),
            sun_body=_unit(sun_body + rng.normal(0.0, self.config.sun_noise_std, 3)),
            sampled_at=t,
        )
        return {"sensors": packet}


class MEKFBlock(Block):
    """Small right-error MEKF inspired by :class:`ADCS.MEKF`."""

    name = "estimator"
    priority = 50
    inputs = {"sensors": SensorPacket, "orbit": np.ndarray}
    outputs = {"estimate": EstimatePacket}

    def __init__(self, timing: Timing) -> None:
        super().__init__(timing)
        self.q = _unit(np.array([0.998, 0.03, -0.04, 0.02]))
        self.bias = np.zeros(3)
        self.p = np.diag(np.r_[np.full(3, 0.04**2), np.full(3, 5e-4**2)])
        self.last_measurement_time: float | None = None

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        sensors: SensorPacket = ctx.get("sensors")
        if self.last_measurement_time is not None and sensors.sampled_at <= self.last_measurement_time + 1e-12:
            return {"estimate": EstimatePacket(self.q, sensors.gyro - self.bias, self.bias, self.p, sensors.sampled_at)}
        dt = self.timing.period if self.last_measurement_time is None else sensors.sampled_at - self.last_measurement_time
        omega = sensors.gyro - self.bias
        self.q = _unit(quat_multiply(self.q, rotvec_quat(omega * max(dt, 0.0))))
        f = np.block([[-skew(omega), -np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])
        phi = np.eye(6) + f * max(dt, 0.0)
        q_process = np.diag(np.r_[np.full(3, 3e-7), np.full(3, 2e-10)]) * max(dt, 1e-6)
        self.p = phi @ self.p @ phi.T + q_process

        orbit = ctx.get("orbit")
        inertial_to_body = quat_to_matrix(self.q).T
        references = ((_unit(magnetic_field_eci(orbit[:3])), _unit(sensors.magnetic_body), 0.012),
                      (_unit(SUN_ECI), _unit(sensors.sun_body), 0.006))
        for reference_eci, measured_body, sigma in references:
            predicted = inertial_to_body @ reference_eci
            residual = measured_body - predicted
            # For the right multiplicative error used here,
            # R(q*dq)^T v ~= R(q)^T v + skew(R(q)^T v) dtheta.
            h = np.c_[skew(predicted), np.zeros((3, 3))]
            s = h @ self.p @ h.T + np.eye(3) * sigma**2
            k = np.linalg.solve(s, h @ self.p).T
            correction = k @ residual
            self.q = _unit(quat_multiply(self.q, rotvec_quat(correction[:3])))
            self.bias += correction[3:]
            ikh = np.eye(6) - k @ h
            self.p = ikh @ self.p @ ikh.T + k @ (np.eye(3) * sigma**2) @ k.T
        self.last_measurement_time = sensors.sampled_at
        return {"estimate": EstimatePacket(self.q, sensors.gyro - self.bias, self.bias, self.p, sensors.sampled_at)}


class ControllerBlock(Block):
    name = "controller"
    priority = 60
    inputs = {"estimate": EstimatePacket, "attitude": np.ndarray, "orbit": np.ndarray}
    outputs = {"control": ControlPacket}

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        estimate: EstimatePacket = ctx.get("estimate")
        attitude = ctx.get("attitude")
        orbit = ctx.get("orbit")
        q_error = estimate.q.copy()
        if q_error[0] < 0:
            q_error = -q_error
        desired_torque = -0.006 * q_error[1:] - 0.025 * estimate.w
        rw = np.clip(desired_torque, -2.0e-3, 2.0e-3)
        h = attitude[7:10]
        b_body = quat_to_matrix(estimate.q).T @ magnetic_field_eci(orbit[:3])
        mtq = 0.06 * np.cross(h, b_body) / max(np.dot(b_body, b_body), 1e-14)
        mtq = np.clip(mtq, -0.20, 0.20)
        return {"control": ControlPacket(mtq, rw, estimate.measurement_time)}


class ActuatorBlock(Block):
    name = "actuators"
    priority = 70
    inputs = {"control": ControlPacket, "orbit": np.ndarray, "attitude": np.ndarray}
    outputs = {"actuator": ActuatorPacket}

    def __init__(self, timing: Timing, config: ModelConfig) -> None:
        super().__init__(timing)
        self.config = config

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        control: ControlPacket = ctx.get("control")
        orbit = ctx.get("orbit")
        attitude = ctx.get("attitude")
        b_body = quat_to_matrix(attitude[3:7]).T @ magnetic_field_eci(orbit[:3])
        mtq_realized = np.clip(control.mtq_dipole, -0.20, 0.20) + rng.normal(0, self.config.mtq_noise_std, 3)
        rw_realized = np.clip(control.rw_torque, -2.0e-3, 2.0e-3) + rng.normal(0, self.config.rw_noise_std, 3)
        torque = np.cross(mtq_realized, b_body) + rw_realized
        return {"actuator": ActuatorPacket(torque, rw_realized, t)}


class LoggerBlock(Block):
    name = "logger"
    priority = 90
    inputs = {
        "orbit": np.ndarray, "attitude": np.ndarray, "sensors": SensorPacket,
        "estimate": EstimatePacket, "control": ControlPacket,
        "actuator": ActuatorPacket, "disturbance": DisturbancePacket,
    }

    def execute(self, t: float, ctx: Context, rng: np.random.Generator) -> dict[str, Any]:
        ctx.append("time", t)
        for key in ("orbit", "attitude", "sensors", "estimate", "control", "actuator", "disturbance"):
            ctx.append(key, ctx.get(key))
            ctx.append(f"{key}_age", t - ctx.signal(key).sampled_at)
        return {}


@dataclass
class SatelliteGraph:
    config: ModelConfig
    blocks: list[Block]
    context: Context

    @property
    def minimum_period(self) -> float:
        return min(block.timing.period for block in self.blocks)

    @property
    def connections(self) -> list[tuple[str, str, str]]:
        producers = {port: block.name for block in self.blocks for port in block.outputs}
        return [
            (producers[port], block.name, port)
            for block in self.blocks for port in block.inputs if port in producers
        ]


def initial_orbit() -> np.ndarray:
    radius = R_EARTH + 500_000.0
    speed = math.sqrt(MU_EARTH / radius)
    inclination = math.radians(51.6)
    return np.array([radius, 0.0, 0.0, 0.0, speed * math.cos(inclination), speed * math.sin(inclination)])


def initial_attitude() -> np.ndarray:
    return np.r_[np.deg2rad([1.2, -0.8, 0.5]), _unit(np.array([0.996, 0.04, -0.055, 0.035])), np.zeros(3)]


def build_satellite(config: ModelConfig) -> SatelliteGraph:
    missing = {"orbit", "attitude", "disturbances", "sensors", "estimator", "controller", "actuators", "logger"} - set(config.timings)
    if missing:
        raise ValueError(f"missing block timings: {sorted(missing)}")
    orbit0 = initial_orbit()
    attitude0 = initial_attitude()
    ctx = Context()
    ctx.put("orbit", orbit0, 0.0, 0.0, "initial")
    ctx.put("attitude", attitude0, 0.0, 0.0, "initial")
    ctx.put("disturbance", DisturbancePacket(np.zeros(3), np.zeros(3)), 0.0, 0.0, "initial")
    inertial_to_body = quat_to_matrix(attitude0[3:7]).T
    initial_sensors = SensorPacket(
        attitude0[:3], inertial_to_body @ magnetic_field_eci(orbit0[:3]), inertial_to_body @ SUN_ECI, 0.0
    )
    ctx.put("sensors", initial_sensors, 0.0, 0.0, "initial")
    initial_estimate = EstimatePacket(attitude0[3:7], attitude0[:3], np.zeros(3), np.eye(6) * 1e-3, 0.0)
    ctx.put("estimate", initial_estimate, 0.0, 0.0, "initial")
    ctx.put("control", ControlPacket(np.zeros(3), np.zeros(3), 0.0), 0.0, 0.0, "initial")
    ctx.put("actuator", ActuatorPacket(np.zeros(3), np.zeros(3), 0.0), 0.0, 0.0, "initial")
    t = config.timings
    blocks: list[Block] = [
        OrbitBlock(t["orbit"], config, orbit0),
        AttitudeBlock(t["attitude"], config, attitude0),
        DisturbanceBlock(t["disturbances"]),
        SensorBlock(t["sensors"], config),
        MEKFBlock(t["estimator"]),
        ControllerBlock(t["controller"]),
        ActuatorBlock(t["actuators"], config),
        LoggerBlock(t["logger"]),
    ]
    producer: dict[str, Block] = {}
    for block in blocks:
        for port, port_type in block.outputs.items():
            if port in producer:
                raise ValueError(f"signal {port!r} has multiple producers: {producer[port].name}, {block.name}")
            producer[port] = block
            if port not in ctx.signals:
                raise ValueError(f"produced signal {port!r} needs an initial value to break feedback cycles")
            if not isinstance(ctx.get(port), port_type):
                raise TypeError(f"initial signal {port!r} does not match {port_type.__name__}")
    for block in blocks:
        for port, port_type in block.inputs.items():
            if port not in ctx.signals:
                raise ValueError(f"block {block.name!r} has unconnected input {port!r}")
            if not isinstance(ctx.get(port), port_type):
                raise TypeError(f"input {block.name}.{port} expects {port_type.__name__}")
    return SatelliteGraph(config, sorted(blocks, key=lambda b: b.priority), ctx)


@dataclass
class SimulationResult:
    config: ModelConfig
    history: dict[str, list[Any]]
    trace: list[TraceEntry]
    backend: str
    wall_time: float
    base_tick: float | None = None
    device: str = "CPU"

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self.history["time"])

    def array(self, name: str) -> np.ndarray:
        values = self.history[name]
        if name == "orbit" or name == "attitude":
            return np.asarray(values)
        if name == "estimate":
            return np.asarray([np.r_[v.w, v.q, v.gyro_bias] for v in values])
        if name == "control":
            return np.asarray([np.r_[v.mtq_dipole, v.rw_torque] for v in values])
        raise KeyError(name)


class FixedStepEngine:
    """Run all clocks on the smallest declared block period."""

    def run(self, graph: SatelliteGraph) -> SimulationResult:
        started = time.perf_counter()
        rng = np.random.default_rng(graph.config.seed)
        base = graph.minimum_period
        for block in graph.blocks:
            period_ratio = block.timing.period / base
            phase_ratio = block.timing.phase / base
            if abs(period_ratio - round(period_ratio)) > 1e-9 or abs(phase_ratio - round(phase_ratio)) > 1e-9:
                raise ValueError(
                    f"fixed-step backend cannot place {block.name!r} on its {base:g}s lattice: "
                    f"period={block.timing.period:g}, phase={block.timing.phase:g}; use AsyncEventEngine"
                )
        ticks = int(math.floor(graph.config.duration / base + 1e-9)) + 1
        for index in range(ticks):
            now = index * base
            for block in graph.blocks:
                relative = now - block.timing.phase
                due = relative >= -1e-12 and abs(relative / block.timing.period - round(relative / block.timing.period)) < 1e-8
                if not due:
                    continue
                outputs = block.execute(now, graph.context, rng)
                graph.context.trace.append(TraceEntry(block.name, now, now, now))
                for name, value in outputs.items():
                    graph.context.put(name, value, now, now, block.name)
        return SimulationResult(
            graph.config, graph.context.history, graph.context.trace, "fixed-minimum-tick",
            time.perf_counter() - started, base_tick=base
        )


class AsyncEventEngine:
    """Priority-queue scheduler with independent clocks and delayed publication."""

    def run(self, graph: SatelliteGraph) -> SimulationResult:
        started = time.perf_counter()
        rng = np.random.default_rng(graph.config.seed)
        queue: list[tuple[float, int, int, str, Any]] = []
        sequence = 0
        for block in graph.blocks:
            heapq.heappush(queue, (block.timing.phase, 1, sequence, "execute", block))
            sequence += 1
        while queue:
            event_time, _, _, kind, payload = heapq.heappop(queue)
            if event_time > graph.config.duration + 1e-12:
                break
            if kind == "deliver":
                name, value, sampled_at, source = payload
                graph.context.put(name, value, sampled_at, event_time, source)
                continue
            block: Block = payload
            outputs = block.execute(event_time, graph.context, rng)
            finish = event_time + block.timing.compute_time
            delivery = finish + block.timing.output_delay
            graph.context.trace.append(TraceEntry(block.name, event_time, finish, delivery))
            for name, value in outputs.items():
                heapq.heappush(queue, (delivery, 0, sequence, "deliver", (name, value, event_time, block.name)))
                sequence += 1
            next_time = event_time + block.timing.period
            heapq.heappush(queue, (next_time, 1, sequence, "execute", block))
            sequence += 1
        return SimulationResult(
            graph.config, graph.context.history, graph.context.trace, "asynchronous-event",
            time.perf_counter() - started
        )


def _cpu_worker(payload: tuple[ModelConfig, int]) -> SimulationResult:
    config, seed = payload
    return FixedStepEngine().run(build_satellite(replace(config, seed=seed)))


def run_cpu_monte_carlo(config: ModelConfig, seeds: Iterable[int], workers: int = 3) -> list[SimulationResult]:
    payloads = [(config, int(seed)) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_cpu_worker, payloads))

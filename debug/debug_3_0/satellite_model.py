"""Satellite blocks built on :mod:`block_engine`.

The equations are intentionally compact; this module demonstrates composition,
clock compilation, nested sensor blocks, and adding a power subsystem without
changing either execution engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .block_engine import Block, CompositeBlock, Model, Port, Recorder
from .prototype import (
    INERTIA,
    MASS,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    SUN_ECI,
    _attitude_integrate,
    _unit,
    initial_attitude,
    initial_orbit,
    magnetic_field_eci,
    orbit_integrate,
    quat_multiply,
    quat_to_matrix,
    rotvec_quat,
)


@dataclass
class Disturbance:
    force_eci: np.ndarray
    torque_body: np.ndarray


@dataclass
class Sensors:
    gyro: np.ndarray
    magnetic_body: np.ndarray
    sun_body: np.ndarray
    sampled_at: float


@dataclass
class Estimate:
    q: np.ndarray
    w: np.ndarray
    bias: np.ndarray


@dataclass
class Command:
    mtq: np.ndarray
    wheels: np.ndarray


@dataclass
class Actuation:
    body_torque: np.ndarray
    wheel_torque: np.ndarray
    power_w: float


@dataclass
class PowerState:
    generated_w: float
    load_w: float
    soc: float
    voltage: float
    actuator_scale: float


ZERO_DISTURBANCE = Disturbance(np.zeros(3), np.zeros(3))
ZERO_ACTUATION = Actuation(np.zeros(3), np.zeros(3), 0.0)
FULL_POWER = PowerState(0.0, 4.0, 0.8, 8.2, 1.0)


class OrbitPropagation(Block):
    def __init__(self, hz: float = 5.0, *, integrator: str = "rk4", degree: int = 4) -> None:
        super().__init__("orbit_propagation", hz, inputs=[Port("disturbance", Disturbance)], outputs=[Port("state", np.ndarray)])
        if integrator not in ("rk2", "rk4"):
            raise ValueError("orbit integrator must be 'rk2' or 'rk4'")
        self.integrator = integrator
        self.degree = degree
        self.state = initial_orbit()

    def initial_outputs(self):
        return {"state": self.state.copy()}

    def step(self, time_s, dt, inputs, rng):
        self.state = orbit_integrate(self.state, dt, inputs["disturbance"].force_eci, self.degree, self.integrator)
        return {"state": self.state.copy()}


class AttitudePropagation(Block):
    def __init__(self, hz: float = 100.0, *, integrator: str = "rk4") -> None:
        super().__init__(
            "attitude_propagation", hz,
            inputs=[Port("actuation", Actuation), Port("disturbance", Disturbance)],
            outputs=[Port("state", np.ndarray)],
        )
        if integrator not in ("rk2", "rk4"):
            raise ValueError("attitude integrator must be 'rk2' or 'rk4'")
        self.order = 2 if integrator == "rk2" else 4
        self.state = initial_attitude()

    def initial_outputs(self):
        return {"state": self.state.copy()}

    def step(self, time_s, dt, inputs, rng):
        actuation = inputs["actuation"]
        torque = actuation.body_torque + inputs["disturbance"].torque_body
        self.state = _attitude_integrate(self.state, dt, torque, actuation.wheel_torque, self.order)
        return {"state": self.state.copy()}


class Disturbances(Block):
    def __init__(self, hz: float = 20.0) -> None:
        super().__init__(
            "disturbances", hz,
            inputs=[Port("orbit", np.ndarray), Port("attitude", np.ndarray)],
            outputs=[Port("disturbance", Disturbance)],
        )

    def initial_outputs(self):
        return {"disturbance": ZERO_DISTURBANCE}

    def step(self, time_s, dt, inputs, rng):
        orbit, attitude = inputs["orbit"], inputs["attitude"]
        r, v, q = orbit[:3], orbit[3:], attitude[3:7]
        rotation = quat_to_matrix(q)
        relative_v = v - np.cross(OMEGA_EARTH, r)
        flow_body = rotation.T @ relative_v
        area = float(np.dot(np.array([0.02, 0.03, 0.06]), np.abs(_unit(flow_body))))
        altitude = np.linalg.norm(r) - R_EARTH
        density = 6e-13 * math.exp(-(altitude - 500_000.0) / 60_000.0)
        force = -0.5 * density * 2.2 * area * np.linalg.norm(relative_v) * relative_v
        drag_torque = np.cross(np.array([0.012, -0.007, 0.018]), rotation.T @ force)
        r_body = rotation.T @ _unit(r)
        gg = 3 * MU_EARTH / np.linalg.norm(r) ** 3 * np.cross(r_body, INERTIA @ r_body)
        return {"disturbance": Disturbance(force, drag_torque + gg)}


class SensorSuite(Block):
    """Monolithic sensor block; safe to sample only when its output is read."""

    def __init__(self, hz: float = 200.0) -> None:
        super().__init__(
            "sensors", hz,
            inputs=[Port("orbit", np.ndarray), Port("attitude", np.ndarray)],
            outputs=[Port("packet", Sensors)],
            sample_on_demand=True,
        )
        self.bias = np.array([2e-4, -1e-4, 1.5e-4])

    def _sample(self, time_s, dt, orbit, attitude, rng):
        self.bias += rng.normal(0.0, 3e-7 * math.sqrt(max(dt, 1e-12)), 3)
        rotation = quat_to_matrix(attitude[3:7]).T
        return Sensors(
            attitude[:3] + self.bias + rng.normal(0.0, 2e-4, 3),
            rotation @ magnetic_field_eci(orbit[:3]) + rng.normal(0.0, 2e-7, 3),
            _unit(rotation @ SUN_ECI + rng.normal(0.0, 2e-3, 3)),
            time_s,
        )

    def initial_outputs(self):
        orbit, attitude = initial_orbit(), initial_attitude()
        rotation = quat_to_matrix(attitude[3:7]).T
        return {"packet": Sensors(attitude[:3], rotation @ magnetic_field_eci(orbit[:3]), rotation @ SUN_ECI, 0.0)}

    def step(self, time_s, dt, inputs, rng):
        return {"packet": self._sample(time_s, dt, inputs["orbit"], inputs["attitude"], rng)}


class Gyro(Block):
    def __init__(self, hz: float = 200.0) -> None:
        super().__init__("gyro", hz, inputs=[Port("attitude", np.ndarray)], outputs=[Port("value", np.ndarray)], sample_on_demand=True)
        self.bias = np.array([2e-4, -1e-4, 1.5e-4])

    def initial_outputs(self):
        return {"value": initial_attitude()[:3].copy()}

    def step(self, time_s, dt, inputs, rng):
        self.bias += rng.normal(0.0, 3e-7 * math.sqrt(max(dt, 1e-12)), 3)
        return {"value": inputs["attitude"][:3] + self.bias + rng.normal(0.0, 2e-4, 3)}


class Magnetometer(Block):
    def __init__(self, hz: float = 20.0) -> None:
        super().__init__(
            "magnetometer", hz,
            inputs=[Port("orbit", np.ndarray), Port("attitude", np.ndarray)],
            outputs=[Port("value", np.ndarray)], sample_on_demand=True,
        )

    def initial_outputs(self):
        orbit, attitude = initial_orbit(), initial_attitude()
        return {"value": quat_to_matrix(attitude[3:7]).T @ magnetic_field_eci(orbit[:3])}

    def step(self, time_s, dt, inputs, rng):
        value = quat_to_matrix(inputs["attitude"][3:7]).T @ magnetic_field_eci(inputs["orbit"][:3])
        return {"value": value + rng.normal(0.0, 2e-7, 3)}


class SunSensor(Block):
    def __init__(self, hz: float = 10.0) -> None:
        super().__init__("sun_sensor", hz, inputs=[Port("attitude", np.ndarray)], outputs=[Port("value", np.ndarray)], sample_on_demand=True)

    def initial_outputs(self):
        return {"value": quat_to_matrix(initial_attitude()[3:7]).T @ SUN_ECI}

    def step(self, time_s, dt, inputs, rng):
        value = quat_to_matrix(inputs["attitude"][3:7]).T @ SUN_ECI
        return {"value": _unit(value + rng.normal(0.0, 2e-3, 3))}


class SensorPacket(Block):
    def __init__(self, hz: float = 200.0) -> None:
        super().__init__(
            "packet", hz,
            inputs=[Port("gyro", np.ndarray), Port("magnetometer", np.ndarray), Port("sun", np.ndarray)],
            outputs=[Port("packet", Sensors)], sample_on_demand=True,
        )

    def initial_outputs(self):
        orbit, attitude = initial_orbit(), initial_attitude()
        rotation = quat_to_matrix(attitude[3:7]).T
        return {"packet": Sensors(attitude[:3], rotation @ magnetic_field_eci(orbit[:3]), rotation @ SUN_ECI, 0.0)}

    def step(self, time_s, dt, inputs, rng):
        return {"packet": Sensors(inputs["gyro"], inputs["magnetometer"], inputs["sun"], time_s)}


class Estimator(Block):
    """Compact multiplicative complementary estimator for architecture tests."""

    def __init__(self, hz: float = 10.0) -> None:
        super().__init__("estimator", hz, inputs=[Port("sensors", Sensors), Port("orbit", np.ndarray)], outputs=[Port("estimate", Estimate)])
        attitude = initial_attitude()
        self.q = attitude[3:7].copy()
        self.bias = np.zeros(3)

    def initial_outputs(self):
        attitude = initial_attitude()
        return {"estimate": Estimate(attitude[3:7].copy(), attitude[:3].copy(), self.bias.copy())}

    def step(self, time_s, dt, inputs, rng):
        sensors, orbit = inputs["sensors"], inputs["orbit"]
        omega = sensors.gyro - self.bias
        self.q = _unit(quat_multiply(self.q, rotvec_quat(omega * dt)))
        rotation = quat_to_matrix(self.q).T
        mag_error = np.cross(_unit(sensors.magnetic_body), _unit(rotation @ magnetic_field_eci(orbit[:3])))
        sun_error = np.cross(_unit(sensors.sun_body), _unit(rotation @ SUN_ECI))
        correction = 0.08 * mag_error + 0.12 * sun_error
        self.q = _unit(quat_multiply(self.q, rotvec_quat(correction)))
        self.bias -= 2e-4 * correction
        return {"estimate": Estimate(self.q.copy(), omega, self.bias.copy())}


class Controller(Block):
    def __init__(self, hz: float = 10.0) -> None:
        super().__init__("controller", hz, inputs=[Port("estimate", Estimate), Port("attitude", np.ndarray), Port("orbit", np.ndarray)], outputs=[Port("command", Command)])

    def initial_outputs(self):
        return {"command": Command(np.zeros(3), np.zeros(3))}

    def step(self, time_s, dt, inputs, rng):
        estimate, attitude, orbit = inputs["estimate"], inputs["attitude"], inputs["orbit"]
        sign = 1.0 if estimate.q[0] >= 0 else -1.0
        wheels = np.clip(-0.006 * sign * estimate.q[1:] - 0.025 * estimate.w, -2e-3, 2e-3)
        b_body = quat_to_matrix(estimate.q).T @ magnetic_field_eci(orbit[:3])
        mtq = np.clip(0.06 * np.cross(attitude[7:10], b_body) / max(np.dot(b_body, b_body), 1e-14), -0.2, 0.2)
        return {"command": Command(mtq, wheels)}


class Actuators(Block):
    def __init__(self, hz: float = 100.0, *, powered: bool = False) -> None:
        ports = [Port("command", Command), Port("orbit", np.ndarray), Port("attitude", np.ndarray)]
        if powered:
            ports.append(Port("power", PowerState))
        super().__init__("actuators", hz, inputs=ports, outputs=[Port("actuation", Actuation), Port("power_draw", float)])
        self.powered = powered

    def initial_outputs(self):
        return {"actuation": ZERO_ACTUATION, "power_draw": 0.0}

    def step(self, time_s, dt, inputs, rng):
        command, orbit, attitude = inputs["command"], inputs["orbit"], inputs["attitude"]
        scale = inputs["power"].actuator_scale if self.powered else 1.0
        wheels = (command.wheels + rng.normal(0.0, 2e-5, 3)) * scale
        mtq = (command.mtq + rng.normal(0.0, 2e-3, 3)) * scale
        b_body = quat_to_matrix(attitude[3:7]).T @ magnetic_field_eci(orbit[:3])
        torque = wheels + np.cross(mtq, b_body)
        power = float(1.2 + 180.0 * np.linalg.norm(wheels) + 2.0 * np.linalg.norm(mtq))
        return {"actuation": Actuation(torque, wheels, power), "power_draw": power}


class SolarPanels(Block):
    def __init__(self, hz: float = 10.0) -> None:
        super().__init__("solar_panels", hz, inputs=[Port("attitude", np.ndarray), Port("orbit", np.ndarray)], outputs=[Port("generated_w", float)], sample_on_demand=True)

    def initial_outputs(self):
        return {"generated_w": 0.0}

    def step(self, time_s, dt, inputs, rng):
        normal_eci = quat_to_matrix(inputs["attitude"][3:7]) @ np.array([1.0, 0.0, 0.0])
        incidence = max(0.0, float(np.dot(normal_eci, _unit(SUN_ECI))))
        # Simple cylindrical eclipse approximation.
        r = inputs["orbit"][:3]
        eclipse = np.dot(r, SUN_ECI) < 0 and np.linalg.norm(r - np.dot(r, SUN_ECI) * SUN_ECI) < R_EARTH
        return {"generated_w": 0.0 if eclipse else float(18.0 * incidence)}


class Battery(Block):
    def __init__(self, hz: float = 10.0, capacity_wh: float = 35.0) -> None:
        super().__init__("battery", hz, inputs=[Port("generated_w", float), Port("actuator_load_w", float)], outputs=[Port("power", PowerState)])
        self.capacity_wh = capacity_wh
        self.soc = 0.8

    def initial_outputs(self):
        return {"power": FULL_POWER}

    def step(self, time_s, dt, inputs, rng):
        generated = inputs["generated_w"]
        load = 4.0 + inputs["actuator_load_w"]
        self.soc = float(np.clip(self.soc + (generated - load) * dt / (3600 * self.capacity_wh), 0.0, 1.0))
        voltage = 6.4 + 2.0 * self.soc
        scale = float(np.clip((voltage - 6.4) / 0.8, 0.0, 1.0))
        return {"power": PowerState(generated, load, self.soc, voltage, scale)}


@dataclass
class SatelliteModel:
    model: Model
    recorder: Recorder
    blocks: dict[str, Block]


def build_model(*, nested_sensors: bool = False, with_power: bool = False) -> SatelliteModel:
    """Construct the same spacecraft with a monolithic or nested sensor block."""

    model = Model("satellite")
    orbit = model.root.add(OrbitPropagation())
    attitude = model.root.add(AttitudePropagation())
    disturbances = model.root.add(Disturbances())

    if nested_sensors:
        sensor_group = model.root.add(CompositeBlock("sensors"))
        gyro = sensor_group.add(Gyro())
        magnetometer = sensor_group.add(Magnetometer())
        sun = sensor_group.add(SunSensor())
        packet = sensor_group.add(SensorPacket())
        sensor_output = packet
        model.connect(attitude, "state", gyro, "attitude")
        model.connect(orbit, "state", magnetometer, "orbit")
        model.connect(attitude, "state", magnetometer, "attitude")
        model.connect(attitude, "state", sun, "attitude")
        model.connect(gyro, "value", packet, "gyro")
        model.connect(magnetometer, "value", packet, "magnetometer")
        model.connect(sun, "value", packet, "sun")
    else:
        sensors = model.root.add(SensorSuite())
        sensor_output = sensors
        model.connect(orbit, "state", sensors, "orbit")
        model.connect(attitude, "state", sensors, "attitude")

    estimator = model.root.add(Estimator())
    controller = model.root.add(Controller())
    actuators = model.root.add(Actuators(powered=with_power))

    model.connect(disturbances, "disturbance", orbit, "disturbance", feedback=True)
    model.connect(actuators, "actuation", attitude, "actuation", feedback=True)
    model.connect(disturbances, "disturbance", attitude, "disturbance", feedback=True)
    model.connect(orbit, "state", disturbances, "orbit")
    model.connect(attitude, "state", disturbances, "attitude")
    model.connect(sensor_output, "packet", estimator, "sensors")
    model.connect(orbit, "state", estimator, "orbit")
    model.connect(estimator, "estimate", controller, "estimate")
    model.connect(attitude, "state", controller, "attitude")
    model.connect(orbit, "state", controller, "orbit")
    model.connect(controller, "command", actuators, "command")
    model.connect(orbit, "state", actuators, "orbit")
    model.connect(attitude, "state", actuators, "attitude")

    blocks: dict[str, Block] = {
        "orbit": orbit, "attitude": attitude, "disturbances": disturbances,
        "sensors": sensor_output, "estimator": estimator,
        "controller": controller, "actuators": actuators,
    }

    recorder_ports = [Port("orbit", np.ndarray), Port("attitude", np.ndarray), Port("estimate", Estimate), Port("actuation", Actuation)]
    if with_power:
        power_group = model.root.add(CompositeBlock("power"))
        solar = power_group.add(SolarPanels())
        battery = power_group.add(Battery())
        model.connect(attitude, "state", solar, "attitude")
        model.connect(orbit, "state", solar, "orbit")
        model.connect(solar, "generated_w", battery, "generated_w")
        model.connect(actuators, "power_draw", battery, "actuator_load_w", feedback=True)
        model.connect(battery, "power", actuators, "power")
        blocks.update({"solar": solar, "battery": battery})
        recorder_ports.append(Port("power", PowerState))

    recorder = model.root.add(Recorder("recorder", 20.0, recorder_ports))
    model.connect(orbit, "state", recorder, "orbit", observer=True)
    model.connect(attitude, "state", recorder, "attitude", observer=True)
    model.connect(estimator, "estimate", recorder, "estimate", observer=True)
    model.connect(actuators, "actuation", recorder, "actuation", observer=True)
    if with_power:
        model.connect(blocks["battery"], "power", recorder, "power", observer=True)
    blocks["recorder"] = recorder
    return SatelliteModel(model, recorder, blocks)

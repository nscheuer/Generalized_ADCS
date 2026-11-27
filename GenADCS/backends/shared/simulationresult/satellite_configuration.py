"""Portable snapshots of the hardware configuration used by a simulation."""

import base64
import pickle

import numpy as np


def _values(value):
    return np.asarray(value, dtype=float).tolist()


def _to_py_goal(goal):
    if goal is None:
        return None
    from GenADCS.backends.python.hardware.ADCS_subsystem import goals as py_goals
    name = type(goal).__name__
    if name in ("NoGoal", "NadirGoal", "ZenithGoal", "AlongTrackGoal", "CrossTrackGoal", "VelocityGoal", "SunGoal"):
        return getattr(py_goals, name)()
    if name == "CoordinateGoal":
        return py_goals.CoordinateGoal(goal.lat_deg, goal.lon_deg, goal.alt_km)
    if name == "CoordinateFMCGoal":
        return py_goals.CoordinateFMCGoal(goal.lat_deg, goal.lon_deg, goal.alt_km, goal.fmc_level)
    if name == "QuaternionGoal":
        return py_goals.QuaternionGoal(goal.quaternion, getattr(goal, "rate", None))
    if name == "AngularRateGoal":
        return py_goals.AngularRateGoal(goal.rate)
    return py_goals.NoGoal()


def _to_py_adcs_sm(adcs_sm):
    from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import ADCSMode, ADCSStateMachine
    modes = [
        ADCSMode(
            name=m.name,
            goal=_to_py_goal(getattr(m, "goal", None)),
            sensors=dict(getattr(m, "sensors", None) or {}),
            actuators=dict(getattr(m, "actuators", None) or {}),
        )
        for m in adcs_sm.modes
    ]
    tl = []
    for t, item in (getattr(adcs_sm, "timeline", None) or ()):
        tl.append((float(t), item if isinstance(item, str) else _to_py_goal(item)))
    return ADCSStateMachine(initial=adcs_sm.initial, modes=modes, timeline=tl)


def _to_py_obc_mode(m):
    from GenADCS.backends.python.hardware.OBC_subsystem.state_machine import Mode
    subs = [_to_py_obc_mode(s) for s in (getattr(m, "submodes", None) or ())]
    raw_b = getattr(m, "bindings", None) or {}
    bindings = {}
    if "adcs" in raw_b and raw_b["adcs"] is not None:
        adcs_b = raw_b["adcs"]
        if isinstance(adcs_b, str):
            bindings["adcs"] = adcs_b
        elif isinstance(adcs_b, tuple) and len(adcs_b) == 2:
            bindings["adcs"] = (adcs_b[0], _to_py_goal(adcs_b[1]))
        else:
            bindings["adcs"] = _to_py_goal(adcs_b)
    if "sensors" in raw_b and raw_b["sensors"] is not None:
        bindings["sensors"] = dict(raw_b["sensors"])
    if "actuators" in raw_b and raw_b["actuators"] is not None:
        bindings["actuators"] = dict(raw_b["actuators"])
    return Mode(
        name=m.name,
        submodes=subs,
        initial_submode=getattr(m, "initial_submode", None),
        duration=getattr(m, "duration", None),
        auto_target=getattr(m, "auto_target", None),
        bindings=bindings,
    )


def _to_py_obc(obc):
    from GenADCS.backends.python.hardware.OBC_subsystem.fdir import AngularRateMonitor, FDIR
    from GenADCS.backends.python.hardware.OBC_subsystem.obc_subsystem import OBCSubsystem
    from GenADCS.backends.python.hardware.OBC_subsystem.state_machine import (
        After, Completion, FDIRTriggered, InFOV, StateMachine, Telecommand, Transition,
    )

    def _conv_tl(raw_tl):
        out = []
        for entry in (raw_tl or ()):
            if len(entry) >= 3 and entry[2] is not None:
                out.append((float(entry[0]), entry[1], _to_py_goal(entry[2])))
            else:
                out.append((float(entry[0]), entry[1]))
        return out

    sm = getattr(obc, "state_machine", None)
    py_sm = None
    if sm is not None:
        modes = [_to_py_obc_mode(m) for m in sm.modes]
        transitions = []
        for tr in sm.transitions:
            trig = tr.trigger
            tname = type(trig).__name__
            if tname == "Telecommand":
                py_trig = Telecommand(
                    trig.command,
                    _to_py_goal(getattr(trig, "goal", None)),
                    dict(trig.sensors) if getattr(trig, "sensors", None) else None,
                    dict(trig.actuators) if getattr(trig, "actuators", None) else None,
                )
            elif tname == "InFOV":
                py_trig = InFOV(
                    trig.lat, trig.lon, trig.alt, trig.max_off_nadir_deg,
                    trig.negate, _to_py_goal(getattr(trig, "goal", None)),
                )
            elif tname == "Completion":
                py_trig = Completion()
            elif tname == "After":
                py_trig = After(trig.duration)
            elif tname == "FDIRTriggered":
                py_trig = FDIRTriggered(getattr(trig, "monitor", None))
            else:
                continue
            transitions.append(Transition(tr.source, tr.target, py_trig, tr.priority))
        py_sm = StateMachine(
            initial=sm.initial, modes=modes, transitions=transitions,
            timeline=_conv_tl(getattr(sm, "timeline", None)),
        )

    fdir = getattr(obc, "fdir", None)
    py_fdir = None
    if fdir is not None:
        mons = [
            AngularRateMonitor(m.max_rate, m.name, getattr(m, "target_mode", None))
            for m in fdir.monitors
        ]
        py_fdir = FDIR(mons, fdir.safe_mode)

    return OBCSubsystem(
        state_machine=py_sm, fdir=py_fdir, N=int(getattr(obc, "N", 1)),
        timeline=_conv_tl(getattr(obc, "timeline", None)),
    )


def _packed(value):
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _unpacked(value):
    return pickle.loads(base64.b64decode(value))


_CPP_SENSOR_ARGS = {
    "Gyroscope": ("position", "orientation", "misalignment", "saturation_value", "noise", "bias"),
    "Magnetometer": ("position", "orientation", "misalignment", "saturation_value", "noise", "bias"),
    "Accelerometer": ("position", "orientation", "misalignment", "saturation_value", "noise", "bias"),
    "CoarseSunSensor": ("position", "orientation", "misalignment", "normal", "sensitivity",
                        "saturation_value", "noise_variance", "bias", "fov_half_angle"),
    "FineSunSensor": ("position", "orientation", "misalignment", "noise", "bias", "fov_half_angles"),
    "FineEarthSensor": ("position", "orientation", "misalignment", "noise", "bias", "fov_half_angles"),
    "GNSS": ("noise", "bias"),
    "AutonomousStarTracker": ("position", "orientation", "misalignment", "saturation_value", "noise", "bias", "fov_half_angle"),
}


def _cpp_sensor_config(sensor):
    kind = type(sensor).__name__
    if kind not in _CPP_SENSOR_ARGS:
        raise TypeError(f"unsupported C++ sensor: {kind}")
    return {"type": kind,
            "args": [_values(getattr(sensor, name)) for name in _CPP_SENSOR_ARGS[kind]],
            "enabled": bool(sensor.enabled), "seed": int(getattr(sensor, "seed", 0))}


def from_satellite(satellite, backend=None):
    """Capture hardware data once, independently of sampled state vectors."""
    geometry = satellite.geometry
    adcs = satellite.adcs
    return {
        "geometry": {
            "mass": float(geometry.mass),
            "inertia": _values(geometry.inertia),
            "center_of_mass": _values(geometry.center_of_mass),
            "panels": [
                {name: _values(getattr(panel, name)) if name in ("position", "orientation", "dimensions")
                 else float(getattr(panel, name))
                 for name in ("position", "orientation", "dimensions", "eta_s", "eta_d", "eta_a", "cd")}
                for panel in geometry.panels
            ],
        },
        "reaction_wheels": [
            ({"legacy_rw": _packed(wheel), "rotational_inertia": float(wheel.J)}
             if hasattr(wheel, "J") else
             {**{name: _values(getattr(wheel, name)) for name in ("position", "orientation", "misalignment")},
              **{name: float(getattr(wheel, name)) for name in ("rotational_inertia", "max_torque", "max_momentum", "noise", "bias")}})
            for wheel in adcs.reaction_wheels
        ],
        "magnetorquers": [
            {**{name: _values(getattr(device, name)) for name in ("position", "orientation", "misalignment")},
             **{name: float(getattr(device, name)) for name in ("max_dipole", "noise", "bias")}}
            for device in adcs.magnetorquers
        ],
        "adcs_N": int(getattr(adcs, "N", 1)),
        "sensors": ([_cpp_sensor_config(sensor) for sensor in adcs.sensors]
                    if backend in ("C", "Rust") or type(satellite).__module__.startswith("GenADCS.backends.cpp")
                    else _packed(adcs.sensors) if getattr(adcs, "sensors", None) else None),
        "estimator": ({"type": "AttitudeMEKF", "args": [_values(getattr(adcs.estimator, name))
                       for name in ("quaternion", "rate", "covariance", "acceleration_psd")],
                       "record": list(adcs.estimator.record)} if backend == "Rust"
                       else _packed(adcs.estimator)) if getattr(adcs, "estimator", None) is not None else None,
        "state_machine": _packed(satellite.state_machine) if satellite.state_machine is not None else None,
        "adcs_state_machine": (
            _packed(adcs.state_machine if backend == "Python" else _to_py_adcs_sm(adcs.state_machine))
            if getattr(adcs, "state_machine", None) is not None
            else None
        ),
        "obc": (
            _packed(satellite.obc if backend == "Python" else _to_py_obc(satellite.obc))
            if getattr(satellite, "obc", None) is not None
            else None
        ),
    }


def to_satellite(config, backend):
    """Rebuild the corresponding backend's satellite from saved hardware data."""
    backend_name = {"Python": "python", "Rust": "rust", "C": "cpp"}[backend]
    hardware = __import__(f"GenADCS.backends.{backend_name}.hardware", fromlist=["Satellite"])
    geometry_data = config["geometry"]
    panels = [hardware.SatellitePanel(**panel) for panel in geometry_data["panels"]]
    geometry = hardware.SatelliteGeometry(
        geometry_data["mass"], np.asarray(geometry_data["inertia"], dtype=float),
        panels, geometry_data["center_of_mass"])
    wheels = [(_unpacked(wheel["legacy_rw"]) if "legacy_rw" in wheel else
              hardware.ReactionWheel(
                  wheel["position"], wheel["orientation"], wheel["rotational_inertia"],
                  wheel["max_torque"], wheel["max_momentum"], wheel["misalignment"],
                  wheel.get("noise", 0.0), wheel.get("bias", 0.0))) for wheel in config["reaction_wheels"]]
    magnetorquers = [hardware.Magnetorquer(
        device["position"], device["orientation"], device["max_dipole"],
        device["misalignment"], device.get("noise", 0.0), device.get("bias", 0.0))
        for device in config["magnetorquers"]]
    adcs = hardware.ADCSSubsystem(wheels, magnetorquers)
    if backend in ("C", "Rust"):
        adcs.N = config.get("adcs_N", 1)
        sensors = []
        for entry in config.get("sensors") or []:
            sensor_module = __import__(f"GenADCS.backends.{backend_name}.hardware.ADCS_subsystem.sensors", fromlist=[entry["type"]])
            sensor = getattr(sensor_module, entry["type"])(*entry["args"])
            sensor.enabled = entry["enabled"]
            if backend == "C":
                sensor.seed = entry["seed"]
            sensors.append(sensor)
        adcs.sensors = sensors
    if backend == "Rust" and config.get("estimator"):
        from GenADCS.backends.rust.hardware.ADCS_subsystem import AttitudeMEKF
        entry = config["estimator"]
        adcs.estimator = AttitudeMEKF(*entry["args"], record=entry["record"])
    if backend == "Python":
        adcs.N = config.get("adcs_N", 1)
        adcs.sensors = _unpacked(config["sensors"]) if config.get("sensors") else []
        adcs.estimator = _unpacked(config["estimator"]) if config.get("estimator") else None
    state_machine = (_unpacked(config["state_machine"])
                     if config["state_machine"] is not None else None)
    return hardware.Satellite(geometry, adcs, state_machine)

"""Portable snapshots of the hardware configuration used by a simulation."""

import base64
import pickle

import numpy as np


def _values(value):
    return np.asarray(value, dtype=float).tolist()


def _packed(value):
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _unpacked(value):
    return pickle.loads(base64.b64decode(value))


def from_satellite(satellite):
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
              **{name: float(getattr(wheel, name)) for name in ("rotational_inertia", "max_torque", "max_momentum")},
              "effects": _packed(wheel.effects)})
            for wheel in adcs.reaction_wheels
        ],
        "magnetorquers": [
            {**{name: _values(getattr(device, name)) for name in ("position", "orientation", "misalignment")},
             "max_dipole": float(device.max_dipole), "effects": _packed(device.effects)}
            for device in adcs.magnetorquers
        ],
        "adcs_N": int(getattr(adcs, "N", 1)),
        "sensors": _packed(adcs.sensors) if getattr(adcs, "sensors", None) else None,
        "estimator": _packed(adcs.estimator) if getattr(adcs, "estimator", None) is not None else None,
        "state_machine": _packed(satellite.state_machine) if satellite.state_machine is not None else None,
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
                  _unpacked(wheel["effects"]))) for wheel in config["reaction_wheels"]]
    magnetorquers = [hardware.Magnetorquer(
        device["position"], device["orientation"], device["max_dipole"],
        device["misalignment"], _unpacked(device["effects"]))
        for device in config["magnetorquers"]]
    adcs = hardware.ADCSSubsystem(wheels, magnetorquers)
    if backend == "Python":
        adcs.N = config.get("adcs_N", 1)
        adcs.sensors = _unpacked(config["sensors"]) if config.get("sensors") else []
        adcs.estimator = _unpacked(config["estimator"]) if config.get("estimator") else None
    state_machine = (_unpacked(config["state_machine"])
                     if config["state_machine"] is not None else None)
    return hardware.Satellite(geometry, adcs, state_machine)

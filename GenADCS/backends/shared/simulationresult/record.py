"""Evaluate requested diagnostics only at recorded trajectory samples."""

import numpy as np
from typing import Any

from .satellite_configuration import from_satellite
from .simulationresult import SatelliteResult, SimulationResult, RESULT_FIELDS, SENSOR_FIELDS


def record_samples(
    times: Any,
    vectors: Any,
    format: Any,
    satellite: Any,
    environment: Any,
    dynamics: Any,
    epoch_jd: float,
    backend: str,
    sensor_samples: dict[str, Any] | None = None,
) -> SimulationResult:
    times = np.asarray(times, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    requested = {name for name in RESULT_FIELDS if getattr(format, name, False)}
    fields = {}
    sensor_requested = requested & SENSOR_FIELDS
    requested -= SENSOR_FIELDS
    if sensor_requested:
        if sensor_samples is None:
            raise ValueError("sensor samples are required for requested sensor fields")
        for name in sensor_requested:
            fields[name] = np.asarray(sensor_samples[name], dtype=np.float64)
    estimate_fields = {"estimated_attitude", "estimated_rate", "estimated_covariance",
                       "estimated_attitude_covariance", "estimated_rate_covariance",
                       "measurement_innovation", "measurement_innovation_covariance",
                       "normalized_innovation_squared", "normalized_estimation_error_squared",
                       "measurement_residual", "kalman_gain"}
    for name in requested & estimate_fields:
        if sensor_samples is None or name not in sensor_samples:
            raise ValueError(f"{name} requires an estimator")
        fields[name] = np.asarray(sensor_samples[name], dtype=np.float64)
    requested -= estimate_fields
    for name in (
        "obc_mode", "adcs_mode", "adcs_goal",
        "goal_attitude", "goal_rate",
        "feedforward_torque", "feedforward_attitude", "feedforward_rate",
        "reference_torque", "commanded_wheel_torque", "commanded_magnetic_dipole",
    ):
        if name in requested:
            if sensor_samples is None or name not in sensor_samples:
                raise ValueError(f"{name} was requested but not recorded by the backend")
            fields[name] = np.asarray(sensor_samples[name], dtype=np.float64)
            requested.remove(name)
    slices = {"satellite_attitude": (0, 4), "satellite_rate": (4, 7),
              "satellite_position": (7, 10), "satellite_velocity": (10, 13),
              "satellite_wheel_momenta": (13, vectors.shape[1])}
    for name, (start, stop) in slices.items():
        if name in requested:
            fields[name] = vectors[:, start:stop].copy()
    diagnostics = requested - slices.keys()
    if diagnostics:
        from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd
        from GenADCS.backends.python.environment.ephemeris import BODY_SUN, eclipse_state
        from GenADCS.backends.python.environment.atmosphere.atmospheric_model import co_rotating_atmosphere_velocity
        from GenADCS.backends.python.dynamics.srp_model.projected_area import SUN_RADIUS_M
        from GenADCS.backends.python.environment.gravity.j_gravity_model import gravity_acceleration_inertial_m_s2
        model_by_name = {type(model).__name__: model for model in dynamics.forces + dynamics.torques}
        body = environment.planetary_body
        mass = float(satellite.geometry.mass)
        mu = float(body.gravitational_parameter)
        radius = float(body.reference_radius)
        output = {name: [] for name in diagnostics}
        sun_fields = {"relative_sun_vector", "eclipse_state"}
        atmosphere_fields = {"atmospheric_density", "relative_atmospheric_velocity"}
        zonal_fields = ("J2_force", "J3_force", "J4_force", "J5_force", "J6_force")
        force_fields = {"gravity_force", *zonal_fields, "lunisolar_force", "aerodynamic_force", "solar_radiation_force", "total_force"}
        torque_fields = {"gravity_gradient_torque", "aerodynamic_torque", "solar_radiation_torque", "magnetic_torque", "total_torque"}
        needed_forces = force_fields if "total_force" in diagnostics else diagnostics & force_fields
        needed_torques = torque_fields if "total_torque" in diagnostics else diagnostics & torque_fields
        force_models = {"aerodynamic_force": "ProjectedAreaDragForceModel",
                        "solar_radiation_force": "ProjectedAreaSRPForceModel",
                        "lunisolar_force": "LunisolarForceModel"}
        torque_models = {"gravity_gradient_torque": "StaticGravityGradientModel",
                         "aerodynamic_torque": "ProjectedAreaDragTorqueModel",
                         "solar_radiation_torque": "ProjectedAreaSRPTorqueModel",
                         "magnetic_torque": "StaticResidualDipoleModel"}
        for time, vector in zip(times, vectors):
            q, rate, pos, vel = vector[:4], vector[4:7], vector[7:10], vector[10:13]
            jd = epoch_jd + time / 86400.0
            values = {}
            if "relative_earth_vector" in diagnostics:
                values["relative_earth_vector"] = -pos
            if diagnostics & sun_fields:
                if environment.ephemeris is None:
                    raise ValueError("Sun diagnostics require an ephemeris")
                sun = 1000.0 * np.asarray(environment.ephemeris.position(BODY_SUN, utc_jd_to_tdb_jd(jd)))
                values["relative_sun_vector"] = sun - pos
                if "eclipse_state" in diagnostics:
                    values["eclipse_state"] = [eclipse_state(pos, sun, radius, SUN_RADIUS_M)[0]]
            if diagnostics & atmosphere_fields:
                if environment.atmosphere is None:
                    raise ValueError("atmospheric diagnostics require an atmosphere")
                values["atmospheric_density"] = [environment.atmosphere.density(pos, jd)]
                values["relative_atmospheric_velocity"] = vel - co_rotating_atmosphere_velocity(pos, jd)
            if "magnetic_field" in diagnostics:
                if environment.magnetic_field is None:
                    raise ValueError("magnetic_field requires an environment magnetic model")
                values["magnetic_field"] = environment.magnetic_field.field_inertial_tesla(pos, jd)
            if needed_forces:
                r = np.linalg.norm(pos)
                central = -mu * mass * pos / r**3
                if "gravity_force" in needed_forces:
                    values["gravity_force"] = central
                zonal = model_by_name.get("ZonalForceModel") if needed_forces.intersection(zonal_fields) else None
                coefficients = np.asarray(zonal.zonal_coefficients) if zonal is not None else np.zeros(5)
                for index, name in enumerate(zonal_fields):
                    if name not in needed_forces:
                        continue
                    if coefficients[index] == 0:
                        values[name] = np.zeros(3)
                    else:
                        selected = np.zeros(5)
                        selected[index] = coefficients[index]
                        values[name] = mass * gravity_acceleration_inertial_m_s2(pos, jd, mu, radius, selected) - central
                for name, model_name in force_models.items():
                    if name not in needed_forces:
                        continue
                    model = model_by_name.get(model_name)
                    if model is None:
                        values[name] = np.zeros(3)
                    elif name == "aerodynamic_force":
                        values[name] = model.evaluate(pos, vel, q, jd, environment)
                    elif name == "solar_radiation_force":
                        values[name] = model.evaluate(pos, q, jd, environment)
                    else:
                        values[name] = model.evaluate(pos, jd)
                if "total_force" in diagnostics:
                    values["total_force"] = sum((values[name] for name in force_fields if name != "total_force"), np.zeros(3))
            if needed_torques:
                for name, model_name in torque_models.items():
                    if name not in needed_torques:
                        continue
                    model = model_by_name.get(model_name)
                    if model is None:
                        values[name] = np.zeros(3)
                    elif name == "gravity_gradient_torque":
                        values[name] = model.evaluate(pos, q, environment)
                    elif name == "aerodynamic_torque":
                        values[name] = model.evaluate(pos, vel, q, jd, environment)
                    else:
                        values[name] = model.evaluate(pos, q, jd, environment)
                if "total_torque" in diagnostics:
                    values["total_torque"] = sum((values[name] for name in torque_fields if name != "total_torque"), np.zeros(3))
            for name in diagnostics:
                output[name].append(values[name])
        fields.update({name: np.asarray(values, dtype=np.float64) for name, values in output.items()})
    sat = SatelliteResult(times, fields, from_satellite(satellite, backend))
    return SimulationResult(epoch_jd, format.N, backend, {"sat_000": sat})

"""Numba RK4 propagation of a passive spacecraft's orbit and attitude."""

from typing import NamedTuple

import numpy as np
from numba import njit

from ADCS_Omni.dynamics.drag_model.projected_area import projected_area_drag_force, projected_area_drag_torque
from ADCS_Omni.dynamics.gravity_gradient_model.static import static_gravity_gradient_torque
from ADCS_Omni.dynamics.residual_dipole_model.static import static_residual_dipole_torque
from ADCS_Omni.dynamics.srp_model.projected_area import (
    SUN_RADIUS_M, projected_area_srp_force, projected_area_srp_torque,
)
from ADCS_Omni.dynamics.third_body_model.lunisolar import (
    MOON_MU_M3_S2, SUN_MU_M3_S2, lunisolar_acceleration,
)
from ADCS_Omni.environment.atmosphere.NRMLSISE00.nrlmsise00 import NRLMSISE00, nrlmsise00_density
from ADCS_Omni.environment.atmosphere.atmospheric_model import co_rotating_atmosphere_velocity
from ADCS_Omni.environment.ephemeris import BODY_MOON, BODY_SUN, ephemeris_position, eclipse_state
from ADCS_Omni.environment.ephemeris.de421 import DE421
from ADCS_Omni.environment.gravity.j_gravity_model import EARTH_ZONAL_COEFFICIENTS, gravity_acceleration_inertial_m_s2
from ADCS_Omni.environment.magnetic_field_model.chaos8.chaos8 import CHAOS8, chaos8_field_inertial_tesla
from ADCS_Omni.environment.magnetic_field_model.igrf14.igrf14 import IGRF14, igrf14_field_inertial_tesla
from ADCS_Omni.helpers.math.derivatives import quaternion_derivative
from ADCS_Omni.helpers.math.time.conversions import utc_jd_to_tdb_jd

SECONDS_PER_DAY = 86400.0


class _Runtime(NamedTuple):
    mu: float
    radius: float
    mass: float
    inverse_inertia: np.ndarray
    inertia: np.ndarray
    zonals: np.ndarray
    drag_force: bool
    drag_torque: bool
    srp_force: bool
    srp_torque: bool
    gravity_gradient: bool
    residual_dipole: bool
    third_body: bool
    normals: np.ndarray
    arms: np.ndarray
    areas: np.ndarray
    drag_coefficients: np.ndarray
    optical_coefficients: np.ndarray
    dipole: np.ndarray
    ephemeris_segments: np.ndarray
    ephemeris_times: np.ndarray
    ephemeris_data: np.ndarray
    magnetic_kind: int
    magnetic_times: np.ndarray
    magnetic_g: np.ndarray
    magnetic_h: np.ndarray
    magnetic_static_g: np.ndarray
    magnetic_static_h: np.ndarray
    magnetic_include_static: bool


@njit(cache=True)
def _derivative(y, epoch_jd, p):
    q = y[0:4].copy()
    q /= np.sqrt(np.dot(q, q))
    omega = y[4:7]
    position = y[7:10]
    velocity = y[10:13]
    force = np.zeros(3)
    torque = np.zeros(3)

    acceleration = gravity_acceleration_inertial_m_s2(position, epoch_jd, p.mu, p.radius, p.zonals)

    if p.drag_force or p.drag_torque:
        density = nrlmsise00_density(position, epoch_jd)
        relative_velocity = velocity - co_rotating_atmosphere_velocity(position, epoch_jd)
        if p.drag_force:
            force += projected_area_drag_force(relative_velocity, q, density, p.normals, p.areas, p.drag_coefficients)
        if p.drag_torque:
            torque += projected_area_drag_torque(relative_velocity, q, density, p.normals, p.arms, p.areas, p.drag_coefficients)

    if p.srp_force or p.srp_torque or p.third_body:
        tdb_jd = utc_jd_to_tdb_jd(epoch_jd)
        sun = 1000.0 * ephemeris_position(BODY_SUN, tdb_jd, p.ephemeris_segments, p.ephemeris_times, p.ephemeris_data)
        if p.third_body:
            moon = 1000.0 * ephemeris_position(BODY_MOON, tdb_jd, p.ephemeris_segments, p.ephemeris_times, p.ephemeris_data)
            acceleration += lunisolar_acceleration(position, sun, moon, SUN_MU_M3_S2, MOON_MU_M3_S2)
        if p.srp_force or p.srp_torque:
            illumination, _, _ = eclipse_state(position, sun, p.radius, SUN_RADIUS_M)
            if p.srp_force:
                force += projected_area_srp_force(position, sun, q, p.normals, p.areas, p.optical_coefficients, illumination)
            if p.srp_torque:
                torque += projected_area_srp_torque(position, sun, q, p.normals, p.arms, p.areas, p.optical_coefficients, illumination)

    if p.gravity_gradient:
        torque += static_gravity_gradient_torque(position, q, p.inertia, p.mu)
    if p.residual_dipole:
        if p.magnetic_kind == 1:
            field = igrf14_field_inertial_tesla(position, epoch_jd, p.magnetic_times, p.magnetic_g, p.magnetic_h)
        else:
            field = chaos8_field_inertial_tesla(position, epoch_jd, p.magnetic_times, p.magnetic_g, p.magnetic_h,
                                                p.magnetic_static_g, p.magnetic_static_h, p.magnetic_include_static)
        torque += static_residual_dipole_torque(p.dipole, field, q)

    acceleration += force / p.mass
    angular_momentum = p.inertia @ omega
    angular_acceleration = p.inverse_inertia @ (torque - np.cross(omega, angular_momentum))
    dy = np.empty(13)
    dy[0:4] = quaternion_derivative(q, omega)
    dy[4:7] = angular_acceleration
    dy[7:10] = velocity
    dy[10:13] = acceleration
    return dy


@njit(cache=True)
def _rk4_step(y, epoch_jd, dt, p):
    k1 = _derivative(y, epoch_jd, p)
    k2 = _derivative(y + 0.5 * dt * k1, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p)
    k3 = _derivative(y + 0.5 * dt * k2, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p)
    k4 = _derivative(y + dt * k3, epoch_jd + dt / SECONDS_PER_DAY, p)
    result = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[0:4] /= np.sqrt(np.dot(result[0:4], result[0:4]))
    return result


def _prepare(satellite, environment, dynamics):
    from ADCS_Omni.dynamics import (
        ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
        ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
        StaticGravityGradientModel, StaticResidualDipoleModel, LunisolarForceModel,
    )
    from ADCS_Omni.dynamics._geometry import prepare_panels

    geometry = satellite.geometry
    mass = float(geometry.mass)
    inertia = np.ascontiguousarray(geometry.inertia, dtype=np.float64)
    if not np.isfinite(mass) or mass <= 0 or inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError("geometry requires positive finite mass and finite 3x3 inertia")
    if not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("geometry.inertia must be symmetric positive definite")
    if satellite.adcs.reaction_wheels:
        raise NotImplementedError("coupled passive propagation does not yet support reaction wheels")
    body = environment.planetary_body
    mu, radius = float(body.gravitational_parameter), float(body.reference_radius)
    if not np.isfinite(mu) or not np.isfinite(radius) or mu <= 0 or radius <= 0:
        raise ValueError("planetary body requires positive finite mu and radius")

    effects = (
        ("drag_force", ProjectedAreaDragForceModel), ("drag_torque", ProjectedAreaDragTorqueModel),
        ("srp_force", ProjectedAreaSRPForceModel), ("srp_torque", ProjectedAreaSRPTorqueModel),
        ("gravity_gradient_torque", StaticGravityGradientModel),
        ("residual_dipole_torque", StaticResidualDipoleModel),
        ("third_body_forces", LunisolarForceModel),
    )
    for flag, model_type in effects:
        if getattr(dynamics, flag):
            model_name = "residual_dipole_model" if flag == "residual_dipole_torque" else "gravity_gradient_model" if flag == "gravity_gradient_torque" else "third_body_model" if flag == "third_body_forces" else flag + "_model"
            if not isinstance(getattr(dynamics, model_name), model_type):
                raise ValueError(f"{flag} requires {model_name} of type {model_type.__name__}")
    has_drag = dynamics.drag_force or dynamics.drag_torque
    has_srp = dynamics.srp_force or dynamics.srp_torque
    if has_drag and not isinstance(environment.atmosphere, NRLMSISE00):
        raise ValueError("compiled drag requires environment.atmosphere=NRLMSISE00()")
    if (has_srp or dynamics.third_body_forces) and not isinstance(environment.ephemeris, DE421):
        raise ValueError("compiled SRP and third-body forces require environment.ephemeris=DE421()")
    if dynamics.residual_dipole_torque and not isinstance(environment.magnetic_field, (IGRF14, CHAOS8)):
        raise ValueError("compiled residual dipole requires IGRF14 or CHAOS8")

    normals, arms, areas, coefficients = prepare_panels(geometry)
    for flag in ("drag_force", "drag_torque", "srp_force", "srp_torque"):
        if getattr(dynamics, flag):
            model = getattr(dynamics, flag + "_model")
            expected = coefficients[:, 0] if flag.startswith("drag") else coefficients[:, 1:4]
            actual = model.drag_coefficients if flag.startswith("drag") else model.optical_coefficients
            if (not np.array_equal(model.normals_b, normals)
                    or not np.array_equal(model.arms_b, arms)
                    or not np.array_equal(model.areas, areas)
                    or not np.array_equal(actual, expected)):
                raise ValueError(f"{flag}_model geometry differs from satellite.geometry")
    if dynamics.gravity_gradient_torque and not np.array_equal(dynamics.gravity_gradient_model.inertia_b, inertia):
        raise ValueError("gravity_gradient_model inertia differs from satellite.geometry")
    if dynamics.third_body_forces:
        model = dynamics.third_body_model
        if model.mass != mass or model.segments is not environment.ephemeris._segments:
            raise ValueError("third_body_model mass or ephemeris differs from the simulation configuration")
    zonals = np.zeros(5)
    configured = environment.j_gravity
    source = configured.zonal_coefficients if configured is not None else EARTH_ZONAL_COEFFICIENTS
    for degree in range(2, 7):
        if getattr(dynamics, f"j{degree}_force"):
            zonals[degree - 2] = source[degree - 2]
    ephemeris = environment.ephemeris
    magnetic = environment.magnetic_field
    magnetic_kind = 1 if isinstance(magnetic, IGRF14) else 2 if isinstance(magnetic, CHAOS8) else 0
    dipole = dynamics.residual_dipole_model.dipole_b if dynamics.residual_dipole_torque else np.zeros(3)
    return _Runtime(
        mu, radius, mass, np.ascontiguousarray(np.linalg.inv(inertia)), inertia, np.ascontiguousarray(zonals),
        dynamics.drag_force, dynamics.drag_torque, dynamics.srp_force, dynamics.srp_torque,
        dynamics.gravity_gradient_torque, dynamics.residual_dipole_torque, dynamics.third_body_forces,
        normals, arms, areas, np.ascontiguousarray(coefficients[:, 0]), np.ascontiguousarray(coefficients[:, 1:4]),
        np.ascontiguousarray(dipole),
        ephemeris._segments if ephemeris is not None else np.empty((0, 5), dtype=np.int64),
        ephemeris._times if ephemeris is not None else np.empty((0, 2)),
        ephemeris._data if ephemeris is not None else np.empty(0),
        magnetic_kind,
        magnetic._epochs if magnetic_kind == 1 else magnetic._times if magnetic_kind == 2 else np.empty(0),
        magnetic._g if magnetic_kind else np.empty((0, 0, 0)),
        magnetic._h if magnetic_kind else np.empty((0, 0, 0)),
        magnetic._static_g if magnetic_kind == 2 else np.empty((0, 0)),
        magnetic._static_h if magnetic_kind == 2 else np.empty((0, 0)),
        magnetic.include_static if magnetic_kind == 2 else False,
    )


class CoupledPropagator:
    """Fixed-step RK4 for Cartesian orbit and body-from-inertial quaternion.

    The state has quaternion, body rates, inertial position and velocity. All
    configured passive forces and torques are reevaluated at every RK stage.
    """

    def __init__(self, dt):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive seconds")
        self._runtime = None
        self._configuration = None

    def reset_runtime(self):
        """Rebuild numerical parameters on the next step after configuration edits."""
        self._runtime = None
        self._configuration = None

    def step(self, state, satellite, environment, dynamics):
        from ADCS_Omni.state import State, AttitudeState, OrbitalState, InternalState

        if self._configuration != (id(satellite), id(environment), id(dynamics)):
            self._runtime = _prepare(satellite, environment, dynamics)
            self._configuration = (id(satellite), id(environment), id(dynamics))
        if state.internal.wheel_momentum.size:
            raise NotImplementedError("coupled passive propagation does not support wheel momentum")
        y = np.concatenate((state.attitude.quaternion_array, state.attitude.rates,
                            state.orbit.position, state.orbit.velocity))
        result = _rk4_step(y, state.orbit.julian_date, self.dt, self._runtime)
        return State(
            AttitudeState(result[0:4], result[4:7]),
            OrbitalState.from_cartesian(result[7:10], result[10:13],
                                        state.orbit.julian_date + self.dt / SECONDS_PER_DAY,
                                        state.orbit.ephemeris),
            InternalState(),
        )

"""Prepared numerical model and compiled coupled equations of motion."""

from typing import NamedTuple

import numpy as np
from numba import njit

from GenADCS.backends.python.dynamics.drag_model.projected_area import (
    projected_area_drag_force,
    projected_area_drag_torque,
)
from GenADCS.backends.python.dynamics.gravity_gradient_model.static import (
    static_gravity_gradient_torque,
)
from GenADCS.backends.python.dynamics.residual_dipole_model.static import (
    static_residual_dipole_torque,
)
from GenADCS.backends.python.dynamics.srp_model.projected_area import (
    SUN_RADIUS_M,
    projected_area_srp_force,
    projected_area_srp_torque,
)
from GenADCS.backends.python.dynamics.third_body_model.lunisolar import (
    MOON_MU_M3_S2,
    SUN_MU_M3_S2,
    lunisolar_acceleration,
)
from GenADCS.backends.python.environment.atmosphere.atmospheric_model import (
    co_rotating_atmosphere_velocity,
)
from GenADCS.backends.python.environment.atmosphere.NRMLSISE00.nrlmsise00 import (
    NRLMSISE00,
    nrlmsise00_density,
)
from GenADCS.backends.python.environment.ephemeris import (
    BODY_MOON,
    BODY_SUN,
    eclipse_state,
    ephemeris_position,
)
from GenADCS.backends.python.environment.ephemeris.de421 import DE421
from GenADCS.backends.python.environment.gravity.j_gravity_model import (
    gravity_acceleration_inertial_m_s2,
)
from GenADCS.backends.python.environment.magnetic_field_model.chaos8.chaos8 import (
    CHAOS8,
    chaos8_field_inertial_tesla,
)
from GenADCS.backends.python.environment.magnetic_field_model.igrf14.igrf14 import (
    IGRF14,
    igrf14_field_inertial_tesla,
)
from GenADCS.backends.python.helpers.math.derivatives import quaternion_derivative
from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd

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
    wheel_axes: np.ndarray
    wheel_inertias: np.ndarray
    inverse_bus_inertia: np.ndarray
    wheel_commands: np.ndarray
    wheel_max_momentum: np.ndarray


@njit(cache=True, inline="always")
def _environment_conditions(position, velocity, epoch_jd, p):
    """Evaluate only the environmental quantities requested by active models.

    Drag force and torque share density/flow; SRP force and torque share the
    Sun and eclipse evaluation. This function runs once per derivative stage.
    """
    density = 0.0
    # Unused contexts alias existing vectors; no placeholder arrays are
    # allocated on every RK stage. Their consumers are guarded by model flags.
    relative_velocity = velocity
    sun = position
    moon = position
    illumination = 0.0
    field = position
    if p.drag_force or p.drag_torque:
        density = nrlmsise00_density(position, epoch_jd)
        relative_velocity = velocity - co_rotating_atmosphere_velocity(
            position, epoch_jd
        )
    if p.srp_force or p.srp_torque or p.third_body:
        tdb_jd = utc_jd_to_tdb_jd(epoch_jd)
        sun = 1000.0 * ephemeris_position(
            BODY_SUN, tdb_jd, p.ephemeris_segments, p.ephemeris_times, p.ephemeris_data
        )
        if p.third_body:
            moon = 1000.0 * ephemeris_position(
                BODY_MOON,
                tdb_jd,
                p.ephemeris_segments,
                p.ephemeris_times,
                p.ephemeris_data,
            )
        if p.srp_force or p.srp_torque:
            illumination, _, _ = eclipse_state(position, sun, p.radius, SUN_RADIUS_M)
    if p.residual_dipole:
        if p.magnetic_kind == 1:
            field = igrf14_field_inertial_tesla(
                position, epoch_jd, p.magnetic_times, p.magnetic_g, p.magnetic_h
            )
        else:
            field = chaos8_field_inertial_tesla(
                position,
                epoch_jd,
                p.magnetic_times,
                p.magnetic_g,
                p.magnetic_h,
                p.magnetic_static_g,
                p.magnetic_static_h,
                p.magnetic_include_static,
            )
    return density, relative_velocity, sun, moon, illumination, field


@njit(cache=True, inline="always")
def _force_acceleration(position, q, epoch_jd, p, conditions):
    """Sum all inertial forces, then divide by spacecraft mass."""
    density, relative_velocity, sun, moon, illumination, _ = conditions
    acceleration = gravity_acceleration_inertial_m_s2(
        position, epoch_jd, p.mu, p.radius, p.zonals
    )
    force = np.zeros(3)
    if p.drag_force:
        force += projected_area_drag_force(
            relative_velocity, q, density, p.normals, p.areas, p.drag_coefficients
        )
    if p.srp_force:
        force += projected_area_srp_force(
            position, sun, q, p.normals, p.areas, p.optical_coefficients, illumination
        )
    if p.third_body:
        acceleration += lunisolar_acceleration(
            position, sun, moon, SUN_MU_M3_S2, MOON_MU_M3_S2
        )
    return acceleration + force / p.mass


@njit(cache=True, inline="always")
def _body_torque(position, q, p, conditions):
    """Sum environmental and actuator torques in body coordinates."""
    density, relative_velocity, sun, _, illumination, field = conditions
    torque = np.zeros(3)
    if p.drag_torque:
        torque += projected_area_drag_torque(
            relative_velocity,
            q,
            density,
            p.normals,
            p.arms,
            p.areas,
            p.drag_coefficients,
        )
    if p.srp_torque:
        torque += projected_area_srp_torque(
            position,
            sun,
            q,
            p.normals,
            p.arms,
            p.areas,
            p.optical_coefficients,
            illumination,
        )
    if p.gravity_gradient:
        torque += static_gravity_gradient_torque(position, q, p.inertia, p.mu)
    if p.residual_dipole:
        torque += static_residual_dipole_torque(p.dipole, field, q)
    for wheel in range(p.wheel_commands.size):
        torque += p.wheel_axes[wheel] * p.wheel_commands[wheel]
    return torque


@njit(cache=True, inline="always")
def _internal_derivative(
    dy, y, q, omega, position, velocity, epoch_jd, conditions, omega_dot, p
):
    """Sum internal providers with access to the full stage and environment.

    ADCS wheel convention: storage torque opposes body torque. Additional
    internal-state providers can fill later slices of this derivative.
    """
    for wheel in range(p.wheel_commands.size):
        momentum_rate = -p.wheel_commands[wheel] - p.wheel_inertias[wheel] * np.dot(
            p.wheel_axes[wheel], omega_dot
        )
        momentum = y[13 + wheel]
        limit = p.wheel_max_momentum[wheel]
        if (momentum >= limit and momentum_rate > 0.0) or (
            momentum <= -limit and momentum_rate < 0.0
        ):
            momentum_rate = 0.0
        dy[13 + wheel] = momentum_rate


@njit(cache=True, inline="always")
def _project_wheel_momentum(y, p):
    """Keep wheel momentum states within their ADCS-defined limits."""
    for wheel in range(p.wheel_commands.size):
        limit = p.wheel_max_momentum[wheel]
        if y[13 + wheel] > limit:
            y[13 + wheel] = limit
        elif y[13 + wheel] < -limit:
            y[13 + wheel] = -limit


@njit(cache=True)
def _derivative(y, epoch_jd, p):
    q = y[0:4].copy()
    q /= np.sqrt(np.dot(q, q))
    omega = y[4:7]
    position = y[7:10]
    velocity = y[10:13]
    conditions = _environment_conditions(position, velocity, epoch_jd, p)
    acceleration = _force_acceleration(position, q, epoch_jd, p, conditions)
    torque = _body_torque(position, q, p, conditions)

    # ADCS/Satellite uses total inertia in H and inertia excluding rotor spin
    # axes in the angular-acceleration solve.
    angular_momentum = p.inertia @ omega
    if p.wheel_axes.shape[0] == 0:
        angular_acceleration = p.inverse_inertia @ (
            torque - np.cross(omega, angular_momentum)
        )
    else:
        for wheel in range(p.wheel_axes.shape[0]):
            angular_momentum += p.wheel_axes[wheel] * y[13 + wheel]
        angular_acceleration = p.inverse_bus_inertia @ (
            torque - np.cross(omega, angular_momentum)
        )
    dy = np.empty(y.size)
    dy[0:4] = quaternion_derivative(q, omega)
    dy[4:7] = angular_acceleration
    dy[7:10] = velocity
    dy[10:13] = acceleration
    if p.wheel_axes.shape[0] != 0:
        _internal_derivative(
            dy,
            y,
            q,
            omega,
            position,
            velocity,
            epoch_jd,
            conditions,
            angular_acceleration,
            p,
        )
    return dy


def _prepare(satellite, environment, dynamics):
    from GenADCS.backends.python.dynamics import (
        LunisolarForceModel,
        ProjectedAreaDragForceModel,
        ProjectedAreaDragTorqueModel,
        ProjectedAreaSRPForceModel,
        ProjectedAreaSRPTorqueModel,
        StaticGravityGradientModel,
        StaticResidualDipoleModel,
        ZonalForceModel,
    )
    from GenADCS.backends.python.dynamics._geometry import prepare_panels

    force_models = dynamics.forces
    torque_models = dynamics.torques
    all_models = force_models + torque_models
    drag_force_model = next(
        (m for m in force_models if isinstance(m, ProjectedAreaDragForceModel)), None
    )
    drag_torque_model = next(
        (m for m in torque_models if isinstance(m, ProjectedAreaDragTorqueModel)), None
    )
    srp_force_model = next(
        (m for m in force_models if isinstance(m, ProjectedAreaSRPForceModel)), None
    )
    srp_torque_model = next(
        (m for m in torque_models if isinstance(m, ProjectedAreaSRPTorqueModel)), None
    )
    gravity_gradient_model = next(
        (m for m in torque_models if isinstance(m, StaticGravityGradientModel)), None
    )
    residual_dipole_model = next(
        (m for m in torque_models if isinstance(m, StaticResidualDipoleModel)), None
    )
    third_body_model = next(
        (m for m in force_models if isinstance(m, LunisolarForceModel)), None
    )
    zonal_model = next(
        (m for m in force_models if isinstance(m, ZonalForceModel)), None
    )
    drag_force, drag_torque = (
        drag_force_model is not None,
        drag_torque_model is not None,
    )
    srp_force, srp_torque = srp_force_model is not None, srp_torque_model is not None
    gravity_gradient = gravity_gradient_model is not None
    residual_dipole = residual_dipole_model is not None
    third_body = third_body_model is not None

    geometry = satellite.geometry
    mass = float(geometry.mass)
    inertia = np.ascontiguousarray(geometry.inertia, dtype=np.float64)
    if (
        not np.isfinite(mass)
        or mass <= 0
        or inertia.shape != (3, 3)
        or not np.all(np.isfinite(inertia))
    ):
        raise ValueError(
            "geometry requires positive finite mass and finite 3x3 inertia"
        )
    if not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("geometry.inertia must be symmetric positive definite")
    # ADCS wheel axes, rotor inertias, momentum and torque sign conventions.
    wheels = satellite.adcs.reaction_wheels
    if wheels:
        from ADCS.satellite_hardware.actuators.reaction_wheel import RW

        if any(not isinstance(wheel, RW) for wheel in wheels):
            raise TypeError(
                "reaction wheels must be ADCS.satellite_hardware.actuators.RW instances"
            )
        if any(bool(wheel.bias) or bool(wheel.noise) for wheel in wheels):
            raise NotImplementedError(
                "compiled wheel torque bias and noise are not yet supported"
            )
    wheel_axes = np.ascontiguousarray(
        np.array([wheel.axis for wheel in wheels], dtype=np.float64).reshape((-1, 3))
    )
    wheel_inertias = np.ascontiguousarray(
        np.array([wheel.J for wheel in wheels], dtype=np.float64)
    )
    wheel_torque_limits = np.asarray(
        [wheel.u_max for wheel in wheels], dtype=np.float64
    )
    wheel_max_momentum = np.asarray(
        [np.max(np.abs(wheel.h_max)) for wheel in wheels], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(wheel_axes))
        or not np.all(np.isfinite(wheel_inertias))
        or np.any(wheel_inertias <= 0.0)
        or not np.all(np.isfinite(wheel_torque_limits))
        or np.any(wheel_torque_limits <= 0.0)
        or not np.all(np.isfinite(wheel_max_momentum))
        or np.any(wheel_max_momentum <= 0.0)
    ):
        raise ValueError("wheel axes, inertias and limits must be finite and valid")
    # The ADCS controller interface is not connected yet. Wheels are passive
    # for now, and every Runge-Kutta stage sees a zero motor torque.
    commands = np.zeros(len(wheels), dtype=np.float64)
    bus_inertia = inertia.copy()
    for wheel in range(len(wheels)):
        bus_inertia -= wheel_inertias[wheel] * np.outer(
            wheel_axes[wheel], wheel_axes[wheel]
        )
    if np.linalg.eigvalsh(bus_inertia).min() <= 0:
        raise ValueError("inertia excluding wheel rotors must be positive definite")
    body = environment.planetary_body
    mu, radius = float(body.gravitational_parameter), float(body.reference_radius)
    if not np.isfinite(mu) or not np.isfinite(radius) or mu <= 0 or radius <= 0:
        raise ValueError("planetary body requires positive finite mu and radius")

    supported = (
        ProjectedAreaDragForceModel,
        ProjectedAreaDragTorqueModel,
        ProjectedAreaSRPForceModel,
        ProjectedAreaSRPTorqueModel,
        StaticGravityGradientModel,
        StaticResidualDipoleModel,
        LunisolarForceModel,
        ZonalForceModel,
    )
    if any(not isinstance(model, supported) for model in all_models):
        raise TypeError("dynamics lists contain an unsupported force or torque model")
    if any(
        isinstance(
            model,
            (
                ProjectedAreaDragForceModel,
                ProjectedAreaSRPForceModel,
                LunisolarForceModel,
                ZonalForceModel,
            ),
        )
        for model in torque_models
    ):
        raise TypeError("force models must be placed in dynamics.forces")
    if any(
        isinstance(
            model,
            (
                ProjectedAreaDragTorqueModel,
                ProjectedAreaSRPTorqueModel,
                StaticGravityGradientModel,
                StaticResidualDipoleModel,
            ),
        )
        for model in force_models
    ):
        raise TypeError("torque models must be placed in dynamics.torques")
    has_drag = drag_force or drag_torque
    has_srp = srp_force or srp_torque
    if has_drag and not isinstance(environment.atmosphere, NRLMSISE00):
        raise ValueError("compiled drag requires environment.atmosphere=NRLMSISE00()")
    if (has_srp or third_body) and not isinstance(environment.ephemeris, DE421):
        raise ValueError(
            "compiled SRP and third-body forces require environment.ephemeris=DE421()"
        )
    if residual_dipole and not isinstance(environment.magnetic_field, (IGRF14, CHAOS8)):
        raise ValueError("compiled residual dipole requires IGRF14 or CHAOS8")

    normals, arms, areas, coefficients = prepare_panels(geometry)
    for flag, model in (
        ("drag_force", drag_force_model),
        ("drag_torque", drag_torque_model),
        ("srp_force", srp_force_model),
        ("srp_torque", srp_torque_model),
    ):
        if model is not None:
            expected = (
                coefficients[:, 0] if flag.startswith("drag") else coefficients[:, 1:4]
            )
            actual = (
                model.drag_coefficients
                if flag.startswith("drag")
                else model.optical_coefficients
            )
            if (
                not np.array_equal(model.normals_b, normals)
                or not np.array_equal(model.arms_b, arms)
                or not np.array_equal(model.areas, areas)
                or not np.array_equal(actual, expected)
            ):
                raise ValueError(
                    f"{flag}_model geometry differs from satellite.geometry"
                )
    if gravity_gradient and not np.array_equal(
        gravity_gradient_model.inertia_b, inertia
    ):
        raise ValueError(
            "gravity_gradient_model inertia differs from satellite.geometry"
        )
    if third_body:
        model = third_body_model
        if model.mass != mass or model.segments is not environment.ephemeris._segments:
            raise ValueError(
                "third_body_model mass or ephemeris differs from the simulation configuration"
            )
    zonals = np.zeros(5)
    if zonal_model is not None:
        zonals[:] = zonal_model.zonal_coefficients
    ephemeris = environment.ephemeris
    magnetic = environment.magnetic_field
    magnetic_kind = (
        1 if isinstance(magnetic, IGRF14) else 2 if isinstance(magnetic, CHAOS8) else 0
    )
    dipole = residual_dipole_model.dipole_b if residual_dipole else np.zeros(3)
    return _Runtime(
        mu,
        radius,
        mass,
        np.ascontiguousarray(np.linalg.inv(inertia)),
        inertia,
        np.ascontiguousarray(zonals),
        drag_force,
        drag_torque,
        srp_force,
        srp_torque,
        gravity_gradient,
        residual_dipole,
        third_body,
        normals,
        arms,
        areas,
        np.ascontiguousarray(coefficients[:, 0]),
        np.ascontiguousarray(coefficients[:, 1:4]),
        np.ascontiguousarray(dipole),
        ephemeris._segments
        if ephemeris is not None
        else np.empty((0, 5), dtype=np.int64),
        ephemeris._times if ephemeris is not None else np.empty((0, 2)),
        ephemeris._data if ephemeris is not None else np.empty(0),
        magnetic_kind,
        magnetic._epochs
        if magnetic_kind == 1
        else magnetic._times
        if magnetic_kind == 2
        else np.empty(0),
        magnetic._g if magnetic_kind else np.empty((0, 0, 0)),
        magnetic._h if magnetic_kind else np.empty((0, 0, 0)),
        magnetic._static_g if magnetic_kind == 2 else np.empty((0, 0)),
        magnetic._static_h if magnetic_kind == 2 else np.empty((0, 0)),
        magnetic.include_static if magnetic_kind == 2 else False,
        wheel_axes,
        wheel_inertias,
        np.ascontiguousarray(np.linalg.inv(bus_inertia)),
        commands,
        np.ascontiguousarray(wheel_max_momentum),
    )

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
from ADCS_Omni.environment.gravity.j_gravity_model import gravity_acceleration_inertial_m_s2
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


@njit(cache=True)
def _integrate_chunk(y, initial_jd, dt, full_steps, remainder, p, output, start, count):
    """Advance consecutive RK4 steps without returning to Python between them."""
    for row in range(start, start + count):
        elapsed_s = row * dt
        step_dt = remainder if row == full_steps else dt
        y = _rk4_step(y, initial_jd + elapsed_s / SECONDS_PER_DAY, step_dt, p)
        output[row + 1, :] = y
    return y


def _prepare(satellite, environment, dynamics):
    from ADCS_Omni.dynamics import (
        ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
        ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
        StaticGravityGradientModel, StaticResidualDipoleModel, LunisolarForceModel,
        ZonalForceModel,
    )
    from ADCS_Omni.dynamics._geometry import prepare_panels

    force_models = dynamics.forces
    torque_models = dynamics.torques
    all_models = force_models + torque_models
    drag_force_model = next((m for m in force_models if isinstance(m, ProjectedAreaDragForceModel)), None)
    drag_torque_model = next((m for m in torque_models if isinstance(m, ProjectedAreaDragTorqueModel)), None)
    srp_force_model = next((m for m in force_models if isinstance(m, ProjectedAreaSRPForceModel)), None)
    srp_torque_model = next((m for m in torque_models if isinstance(m, ProjectedAreaSRPTorqueModel)), None)
    gravity_gradient_model = next((m for m in torque_models if isinstance(m, StaticGravityGradientModel)), None)
    residual_dipole_model = next((m for m in torque_models if isinstance(m, StaticResidualDipoleModel)), None)
    third_body_model = next((m for m in force_models if isinstance(m, LunisolarForceModel)), None)
    zonal_model = next((m for m in force_models if isinstance(m, ZonalForceModel)), None)
    drag_force, drag_torque = drag_force_model is not None, drag_torque_model is not None
    srp_force, srp_torque = srp_force_model is not None, srp_torque_model is not None
    gravity_gradient = gravity_gradient_model is not None
    residual_dipole = residual_dipole_model is not None
    third_body = third_body_model is not None

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

    supported = (ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
                 ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
                 StaticGravityGradientModel, StaticResidualDipoleModel,
                 LunisolarForceModel, ZonalForceModel)
    if any(not isinstance(model, supported) for model in all_models):
        raise TypeError("dynamics lists contain an unsupported force or torque model")
    if any(isinstance(model, (ProjectedAreaDragForceModel, ProjectedAreaSRPForceModel,
                              LunisolarForceModel, ZonalForceModel)) for model in torque_models):
        raise TypeError("force models must be placed in dynamics.forces")
    if any(isinstance(model, (ProjectedAreaDragTorqueModel, ProjectedAreaSRPTorqueModel,
                              StaticGravityGradientModel, StaticResidualDipoleModel)) for model in force_models):
        raise TypeError("torque models must be placed in dynamics.torques")
    has_drag = drag_force or drag_torque
    has_srp = srp_force or srp_torque
    if has_drag and not isinstance(environment.atmosphere, NRLMSISE00):
        raise ValueError("compiled drag requires environment.atmosphere=NRLMSISE00()")
    if (has_srp or third_body) and not isinstance(environment.ephemeris, DE421):
        raise ValueError("compiled SRP and third-body forces require environment.ephemeris=DE421()")
    if residual_dipole and not isinstance(environment.magnetic_field, (IGRF14, CHAOS8)):
        raise ValueError("compiled residual dipole requires IGRF14 or CHAOS8")

    normals, arms, areas, coefficients = prepare_panels(geometry)
    for flag, model in (("drag_force", drag_force_model), ("drag_torque", drag_torque_model),
                        ("srp_force", srp_force_model), ("srp_torque", srp_torque_model)):
        if model is not None:
            expected = coefficients[:, 0] if flag.startswith("drag") else coefficients[:, 1:4]
            actual = model.drag_coefficients if flag.startswith("drag") else model.optical_coefficients
            if (not np.array_equal(model.normals_b, normals)
                    or not np.array_equal(model.arms_b, arms)
                    or not np.array_equal(model.areas, areas)
                    or not np.array_equal(actual, expected)):
                raise ValueError(f"{flag}_model geometry differs from satellite.geometry")
    if gravity_gradient and not np.array_equal(gravity_gradient_model.inertia_b, inertia):
        raise ValueError("gravity_gradient_model inertia differs from satellite.geometry")
    if third_body:
        model = third_body_model
        if model.mass != mass or model.segments is not environment.ephemeris._segments:
            raise ValueError("third_body_model mass or ephemeris differs from the simulation configuration")
    zonals = np.zeros(5)
    if zonal_model is not None:
        zonals[:] = zonal_model.zonal_coefficients
    ephemeris = environment.ephemeris
    magnetic = environment.magnetic_field
    magnetic_kind = 1 if isinstance(magnetic, IGRF14) else 2 if isinstance(magnetic, CHAOS8) else 0
    dipole = residual_dipole_model.dipole_b if residual_dipole else np.zeros(3)
    return _Runtime(
        mu, radius, mass, np.ascontiguousarray(np.linalg.inv(inertia)), inertia, np.ascontiguousarray(zonals),
        drag_force, drag_torque, srp_force, srp_torque,
        gravity_gradient, residual_dipole, third_body,
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


class CoupledRK4Propagator:
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

    def integrate(self, state, satellite, environment, dynamics, tf, progress=False,
                  progress_chunk_steps=500):
        """Return elapsed seconds and numeric states for the complete interval.

        Numeric state columns are quaternion (0:4), body rates (4:7),
        inertial position (7:10), and inertial velocity (10:13).
        """
        tf = float(tf)
        if not np.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if progress and (isinstance(progress_chunk_steps, bool)
                         or not isinstance(progress_chunk_steps, int)
                         or progress_chunk_steps <= 0):
            raise ValueError("progress_chunk_steps must be a positive integer")
        if state.internal.wheel_momentum.size:
            raise NotImplementedError("coupled passive propagation does not support wheel momentum")

        runtime = _prepare(satellite, environment, dynamics)
        y = np.concatenate((state.attitude.quaternion_array, state.attitude.rates,
                            state.orbit.position, state.orbit.velocity))
        full_steps = int(np.floor(tf / self.dt))
        remainder = tf - full_steps * self.dt
        if remainder <= 1e-12 * max(1.0, tf):
            remainder = 0.0
        step_count = full_steps + int(remainder > 0.0)
        elapsed = np.minimum(np.arange(step_count + 1, dtype=np.float64) * self.dt, tf)
        elapsed[-1] = tf
        output = np.empty((step_count + 1, 13), dtype=np.float64)
        output[0, :] = y

        if progress and step_count:
            from tqdm import tqdm

            with tqdm(total=1, desc="Compiling Numba kernels", unit="stage") as bar:
                _integrate_chunk(y, state.orbit.julian_date, self.dt, full_steps,
                                 remainder, runtime, output, 0, 0)
                bar.update(1)
            with tqdm(total=step_count, desc="Running coupled RK4", unit="step") as bar:
                for start in range(0, step_count, progress_chunk_steps):
                    count = min(progress_chunk_steps, step_count - start)
                    y = _integrate_chunk(y, state.orbit.julian_date, self.dt,
                                         full_steps, remainder, runtime, output,
                                         start, count)
                    bar.update(count)
        elif step_count:
            _integrate_chunk(y, state.orbit.julian_date, self.dt, full_steps,
                             remainder, runtime, output, 0, step_count)
        return elapsed, output

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

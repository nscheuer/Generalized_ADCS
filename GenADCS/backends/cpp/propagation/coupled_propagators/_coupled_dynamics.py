"""Prepare C backend model data for the native coupled integrators."""
from types import SimpleNamespace
import numpy as np
from GenADCS.backends.cpp.dynamics import (
    ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
    StaticGravityGradientModel, StaticResidualDipoleModel,
    LunisolarForceModel, ZonalForceModel,
)
from GenADCS.backends.cpp.dynamics._geometry import prepare_panels

SECONDS_PER_DAY = 86400.0

def _prepare(satellite, environment, dynamics):
    forces = dynamics.forces
    torques = dynamics.torques
    geometry = satellite.geometry
    mass = float(geometry.mass)
    inertia = np.ascontiguousarray(geometry.inertia, dtype=np.float64)
    if not np.isfinite(mass) or mass <= 0 or inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError("geometry requires positive finite mass and finite 3x3 inertia")
    if not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("geometry.inertia must be symmetric positive definite")
    wheels = satellite.adcs.reaction_wheels
    if wheels:
        raise NotImplementedError("C coupled propagation with reaction wheels is not implemented")
    body = environment.planetary_body
    mu, radius = float(body.gravitational_parameter), float(body.reference_radius)
    if not np.isfinite(mu) or not np.isfinite(radius) or mu <= 0 or radius <= 0:
        raise ValueError("planetary body requires positive finite mu and radius")
    supported_forces = (ZonalForceModel, ProjectedAreaDragForceModel,
                        ProjectedAreaSRPForceModel, LunisolarForceModel)
    supported_torques = (StaticGravityGradientModel, ProjectedAreaDragTorqueModel,
                         ProjectedAreaSRPTorqueModel, StaticResidualDipoleModel)
    if any(not isinstance(model, supported_forces) for model in forces):
        raise TypeError("dynamics.forces contains an unsupported force model")
    if any(not isinstance(model, supported_torques) for model in torques):
        raise TypeError("dynamics.torques contains an unsupported torque model")
    pick = lambda models, cls: next((m for m in models if isinstance(m, cls)), None)
    df = pick(forces, ProjectedAreaDragForceModel)
    dt = pick(torques, ProjectedAreaDragTorqueModel)
    sf = pick(forces, ProjectedAreaSRPForceModel)
    st = pick(torques, ProjectedAreaSRPTorqueModel)
    gg = pick(torques, StaticGravityGradientModel)
    rd = pick(torques, StaticResidualDipoleModel)
    tb = pick(forces, LunisolarForceModel)
    zm = pick(forces, ZonalForceModel)
    if (df or dt) and environment.atmosphere is None:
        raise ValueError("drag requires environment.atmosphere")
    if (sf or st or tb) and environment.ephemeris is None:
        raise ValueError("SRP and third-body models require environment.ephemeris")
    if rd and environment.magnetic_field is None:
        raise ValueError("residual dipole requires environment.magnetic_field")
    normals, arms, areas, coefficients = prepare_panels(geometry)
    if gg and not np.array_equal(gg.inertia_b, inertia):
        raise ValueError("gravity gradient inertia differs from satellite.geometry")
    return SimpleNamespace(
        mu=mu, radius=radius, mass=mass, inertia=inertia,
        inverse_inertia=np.linalg.inv(inertia), inverse_bus_inertia=np.linalg.inv(inertia),
        zonals=np.ascontiguousarray(zm.zonal_coefficients if zm else np.zeros(5)),
        drag_force=bool(df), drag_torque=bool(dt), srp_force=bool(sf), srp_torque=bool(st),
        gravity_gradient=bool(gg), residual_dipole=bool(rd), third_body=bool(tb),
        normals=normals, arms=arms, areas=areas,
        drag_coefficients=np.ascontiguousarray(coefficients[:, 0]),
        optical_coefficients=np.ascontiguousarray(coefficients[:, 1:4]),
        dipole=np.ascontiguousarray(rd.dipole_b if rd else np.zeros(3)),
        magnetic_kind=1 if environment.magnetic_field is not None else 0,
        magnetic_include_static=bool(getattr(environment.magnetic_field, "include_static", True)),
        wheel_axes=np.empty((0, 3)), wheel_inertias=np.empty(0),
        wheel_commands=np.empty(0), wheel_max_momentum=np.empty(0),
    )

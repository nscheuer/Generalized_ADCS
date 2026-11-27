"""Prepare the fixed model data used by native coupled integration."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from GenADCS.backends.rust.dynamics import (
    ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
    StaticGravityGradientModel, StaticResidualDipoleModel,
    LunisolarForceModel, ZonalForceModel,
)
from GenADCS.backends.rust.dynamics._geometry import prepare_panels
SECONDS_PER_DAY=86400.0


def _prepare(satellite, environment, dynamics):
    geometry=satellite.geometry
    mass=float(geometry.mass)
    inertia=np.ascontiguousarray(geometry.inertia,dtype=np.float64)
    if not np.isfinite(mass) or mass<=0 or inertia.shape!=(3,3) or not np.all(np.isfinite(inertia)):
        raise ValueError("geometry requires positive finite mass and finite 3x3 inertia")
    if not np.allclose(inertia,inertia.T) or np.linalg.eigvalsh(inertia).min()<=0:
        raise ValueError("geometry.inertia must be symmetric positive definite")
    wheels = satellite.adcs.reaction_wheels
    if any(wheel.bias or wheel.noise for wheel in wheels):
        raise NotImplementedError("compiled wheel torque bias and noise are not yet supported")
    wheel_axes=np.ascontiguousarray(np.array([wheel.orientation for wheel in wheels],dtype=np.float64).reshape((-1,3)))
    wheel_inertias=np.ascontiguousarray(np.array([wheel.rotational_inertia for wheel in wheels],dtype=np.float64))
    wheel_torque_limits=np.asarray([wheel.max_torque for wheel in wheels],dtype=np.float64)
    wheel_max_momentum=np.asarray([wheel.max_momentum for wheel in wheels],dtype=np.float64)
    if (
        not np.all(np.isfinite(wheel_axes))
        or not np.all(np.isfinite(wheel_inertias))
        or np.any(wheel_inertias<=0.0)
        or not np.all(np.isfinite(wheel_torque_limits))
        or np.any(wheel_torque_limits<=0.0)
        or not np.all(np.isfinite(wheel_max_momentum))
        or np.any(wheel_max_momentum<=0.0)
    ):
        raise ValueError("wheel axes, inertias and limits must be finite and valid")
    commands=np.zeros(len(wheels),dtype=np.float64)
    bus_inertia=inertia.copy()
    for wheel in range(len(wheels)):
        bus_inertia-=wheel_inertias[wheel]*np.outer(wheel_axes[wheel],wheel_axes[wheel])
    if np.linalg.eigvalsh(bus_inertia).min()<=0:
        raise ValueError("inertia excluding wheel rotors must be positive definite")
    inverse_bus_inertia=np.ascontiguousarray(np.linalg.inv(bus_inertia))
    body=environment.planetary_body
    mu,radius=float(body.gravitational_parameter),float(body.reference_radius)
    if not np.isfinite(mu) or not np.isfinite(radius) or mu<=0 or radius<=0:
        raise ValueError("planetary body requires positive finite mu and radius")
    forces=tuple(dynamics.forces)
    torques=tuple(dynamics.torques)
    force_types=(ZonalForceModel,ProjectedAreaDragForceModel,ProjectedAreaSRPForceModel,LunisolarForceModel)
    torque_types=(StaticGravityGradientModel,ProjectedAreaDragTorqueModel,ProjectedAreaSRPTorqueModel,StaticResidualDipoleModel)
    if any(not isinstance(model,force_types) for model in forces):
        raise TypeError("dynamics.forces contains an unsupported force model")
    if any(not isinstance(model,torque_types) for model in torques):
        raise TypeError("dynamics.torques contains an unsupported torque model")
    if any(isinstance(m,(ProjectedAreaDragForceModel,ProjectedAreaDragTorqueModel)) for m in forces+torques) and environment.atmosphere is None:
        raise ValueError("drag requires environment.atmosphere")
    if any(isinstance(m,(ProjectedAreaSRPForceModel,ProjectedAreaSRPTorqueModel,LunisolarForceModel)) for m in forces+torques) and environment.ephemeris is None:
        raise ValueError("SRP and third-body models require environment.ephemeris")
    if any(isinstance(m,StaticResidualDipoleModel) for m in torques) and environment.magnetic_field is None:
        raise ValueError("residual dipole requires environment.magnetic_field")
    if any(isinstance(m,StaticGravityGradientModel) and not np.array_equal(m.inertia_b,inertia) for m in torques):
        raise ValueError("gravity gradient inertia differs from satellite.geometry")
    pick=lambda models,cls:next((m for m in models if isinstance(m,cls)),None)
    zm=pick(forces,ZonalForceModel)
    rd=pick(torques,StaticResidualDipoleModel)
    normals,arms,areas,coefficients=prepare_panels(geometry)
    magnetic=environment.magnetic_field
    mtqs = satellite.adcs.magnetorquers
    mtq_axes = np.ascontiguousarray(np.array([m.orientation for m in mtqs], dtype=np.float64).reshape((-1, 3)))
    mtq_max_dipole = np.ascontiguousarray(np.array([m.max_dipole for m in mtqs], dtype=np.float64))
    dipole_commands = np.zeros(len(mtqs), dtype=np.float64)
    return SimpleNamespace(
        mass=mass,mu=mu,radius=radius,inertia=inertia,inverse_inertia=np.linalg.inv(inertia),
        zonals=np.ascontiguousarray(zm.zonal_coefficients if zm else np.zeros(5)),
        normals=normals,arms=arms,areas=areas,
        drag_coefficients=np.ascontiguousarray(coefficients[:,0]),
        optical_coefficients=np.ascontiguousarray(coefficients[:,1:4]),
        dipole=np.ascontiguousarray(rd.dipole_b if rd else np.zeros(3)),
        drag_force=bool(pick(forces,ProjectedAreaDragForceModel)),
        drag_torque=bool(pick(torques,ProjectedAreaDragTorqueModel)),
        srp_force=bool(pick(forces,ProjectedAreaSRPForceModel)),
        srp_torque=bool(pick(torques,ProjectedAreaSRPTorqueModel)),
        gravity_gradient=bool(pick(torques,StaticGravityGradientModel)),
        residual_dipole=bool(rd),third_body=bool(pick(forces,LunisolarForceModel)),
        magnetic_kind=1 if type(magnetic).__name__=="IGRF14" else 2 if type(magnetic).__name__=="CHAOS8" else 0,
        magnetic_include_static=bool(getattr(magnetic,"include_static",True)),
        ephemeris_path=str(Path(__file__).resolve().parents[4] / "kernels/environment/ephemeris/de421/de421.bsp"),
        wheel_axes=wheel_axes,wheel_inertias=wheel_inertias,inverse_bus_inertia=inverse_bus_inertia,
        wheel_commands=commands,wheel_max_momentum=np.ascontiguousarray(wheel_max_momentum),
        mtq_axes=mtq_axes,mtq_max_dipole=mtq_max_dipole,dipole_commands=dipole_commands,
    )

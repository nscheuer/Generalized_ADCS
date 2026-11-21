"""Prepare the fixed model data used by native coupled integration."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from ADCS_Omni.ADCS_Rust.dynamics import (
    ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
    StaticGravityGradientModel, StaticResidualDipoleModel,
    LunisolarForceModel, ZonalForceModel,
)
from ADCS_Omni.ADCS_Rust.dynamics._geometry import prepare_panels
from ADCS_Omni.ADCS_Rust.environment.ephemeris.de421 import de421 as de421_module
SECONDS_PER_DAY=86400.0


def _prepare(satellite, environment, dynamics):
    geometry=satellite.geometry
    mass=float(geometry.mass)
    inertia=np.ascontiguousarray(geometry.inertia,dtype=np.float64)
    if not np.isfinite(mass) or mass<=0 or inertia.shape!=(3,3) or not np.all(np.isfinite(inertia)):
        raise ValueError("geometry requires positive finite mass and finite 3x3 inertia")
    if not np.allclose(inertia,inertia.T) or np.linalg.eigvalsh(inertia).min()<=0:
        raise ValueError("geometry.inertia must be symmetric positive definite")
    if satellite.adcs.reaction_wheels:
        raise NotImplementedError("Rust coupled propagation with reaction wheels is not implemented")
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
        ephemeris_path=str(Path(de421_module.__file__).with_name("de421.bsp")),
        wheel_commands=np.empty(0),wheel_max_momentum=np.empty(0),
    )

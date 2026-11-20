from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ADCS_Omni.dynamics.drag_model import ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel
    from ADCS_Omni.dynamics.srp_model import ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel
    from ADCS_Omni.dynamics.gravity_gradient_model import StaticGravityGradientModel
    from ADCS_Omni.dynamics.residual_dipole_model import StaticResidualDipoleModel
    from ADCS_Omni.dynamics.third_body_model import LunisolarForceModel


@dataclass(slots=True)
class DynamicsConfig:
    drag_torque: bool = False
    drag_torque_model: ProjectedAreaDragTorqueModel | None = None
    srp_torque: bool = False
    srp_torque_model: ProjectedAreaSRPTorqueModel | None = None
    residual_dipole_torque: bool = False
    residual_dipole_model: StaticResidualDipoleModel | None = None
    gravity_gradient_torque: bool = False
    gravity_gradient_model: StaticGravityGradientModel | None = None
    drag_force: bool = False
    drag_force_model: ProjectedAreaDragForceModel | None = None
    srp_force: bool = False
    srp_force_model: ProjectedAreaSRPForceModel | None = None
    j2_force: bool = False
    j3_force: bool = False
    j4_force: bool = False
    j5_force: bool = False
    j6_force: bool = False
    third_body_forces: bool = False
    third_body_model: LunisolarForceModel | None = None

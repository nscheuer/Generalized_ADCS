from dataclasses import dataclass


@dataclass(slots=True)
class DynamicsConfig:
    drag_torque: bool = False
    drag_torque_model: object | None = None
    srp_torque: bool = False
    srp_torque_model: object | None = None
    residual_dipole_torque: bool = False
    residual_dipole_model: object | None = None
    gravity_gradient_torque: bool = False
    gravity_gradient_model: object | None = None
    drag_force: bool = False
    drag_force_model: object | None = None
    srp_force: bool = False
    srp_force_model: object | None = None
    j2_force: bool = False
    j3_force: bool = False
    j4_force: bool = False
    j5_force: bool = False
    j6_force: bool = False
    third_body_forces: bool = False
    third_body_model: object | None = None

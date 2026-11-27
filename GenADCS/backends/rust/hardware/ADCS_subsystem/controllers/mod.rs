pub mod allocator;
pub mod attitude_controller;
pub mod bcross;
pub mod bdot;
pub mod controller;
pub mod controllers_py;
pub mod desaturation;
pub mod feedback;
pub mod feedforward;
pub mod runtime;

pub use allocator::{
    allocate_l2, allocate_linfinity, saturate_wheels, Allocator, L2Allocator, LInfinityAllocator,
    ALLOC_L2, ALLOC_LINFINITY,
};
pub use attitude_controller::AttitudeController;
pub use bcross::BCross;
pub use bdot::BDot;
pub use controller::{
    allocate_dipoles, read_first_sensor_body, Controller, NoController, SensorConfig,
    CTRL_ATTITUDE, CTRL_BCROSS, CTRL_BDOT, CTRL_NONE,
};
pub use desaturation::{
    hogan_schaub_mtq_dump, hogan_schaub_u0, Desaturation, HoganSchaubManagement, NoDesaturation,
    DESAT_HOGAN_SCHAUB, DESAT_NONE,
};
pub use feedback::{
    mrp_feedback_linearized_step, to_gain_matrix, Feedback, GainInput, MRPFeedbackLinearized, Mat3,
    FB_MRP_FEEDBACK_LINEARIZED,
};
pub use feedforward::{passthrough_step, Feedforward, Passthrough, FF_PASSTHROUGH};
pub use runtime::{
    init_controller_work, prepare_controllers, step_controller, sync_actuator_enable,
    ControllerRuntime, ControllerTable, CTRL_PARAM_WIDTH,
};

#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/test_bdot.rs"]
mod test_bdot;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/test_bcross.rs"]
mod test_bcross;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/test_attitudecontroller.rs"]
mod test_attitudecontroller;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/feedforward/test_passthrough.rs"]
mod test_passthrough;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/feedback/test_mrpfeedbacklinearization.rs"]
mod test_mrpfeedbacklinearization;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/desaturation/test_hoganschaubdesaturation.rs"]
mod test_hoganschaubdesaturation;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/allocators/test_l2allocator.rs"]
mod test_l2allocator;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/controllers/allocators/test_linftyallocator.rs"]
mod test_linftyallocator;

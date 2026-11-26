#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod effects;
mod effects_py;
pub use effects::{apply_sequence, Effect};
use effects_py::apply_effects;
mod adcs_subsystem;
mod adcs_subsystem_py;
pub mod estimator;
pub mod sensors;
pub use adcs_subsystem::ADCSSubsystem as NativeADCSSubsystem;
use adcs_subsystem_py::ADCSSubsystem;
pub use estimator::attitude_estimator::NativeAttitudeEKF;
pub use estimator::attitude_estimator::NativeAttitudeEstimator;
pub use estimator::attitude_estimator::NativeAttitudeMEKF;
pub use sensors::Sensor as NativeSensor;

include!("magnetorquer.rs");
include!("magnetorquer_py.rs");
include!("reaction_wheel.rs");
include!("reaction_wheel_py.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_adcs_subsystem.rs"]
mod test_adcs_subsystem;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_effects.rs"]
mod test_effects;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_magnetorquer.rs"]
mod test_magnetorquer;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_reaction_wheel.rs"]
mod test_reaction_wheel;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_sensor.rs"]
mod test_sensor;

#[pymodule]
fn _subsystem(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_effects, m)?)?;
    sensors::register(m)?;
    estimator::attitude_estimator::register(m)?;
    m.add_class::<Magnetorquer>()?;
    m.add_class::<ReactionWheel>()?;
    m.add_class::<ADCSSubsystem>()?;
    Ok(())
}

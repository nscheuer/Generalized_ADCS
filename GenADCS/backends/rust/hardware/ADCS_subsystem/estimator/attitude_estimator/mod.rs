pub mod attitude_mekf;
mod attitude_mekf_py;
pub mod base;
mod base_py;

pub use attitude_mekf::AttitudeMEKF as NativeAttitudeMEKF;
pub use base::AttitudeEstimator;
pub use base::AttitudeEstimator as NativeAttitudeEstimator;
pub use base_py::PyAttitudeEstimator;

#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_mekf.rs"]
mod test_attitude_mekf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_base.rs"]
mod test_base;

pub fn register(module: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    attitude_mekf_py::register(module)
}

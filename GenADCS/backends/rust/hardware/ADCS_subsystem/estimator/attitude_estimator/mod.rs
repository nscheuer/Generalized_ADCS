pub mod attitude_batched_ukf;
mod attitude_batched_ukf_py;
pub mod attitude_ekf;
mod attitude_ekf_py;
pub mod attitude_mekf;
mod attitude_mekf_py;
pub mod attitude_sequential_srukf;
mod attitude_sequential_srukf_py;
pub mod attitude_sequential_ukf;
mod attitude_sequential_ukf_py;
pub mod base;
mod base_py;
pub mod direction_correction;

pub use attitude_batched_ukf::AttitudeBatchedUKF as NativeAttitudeBatchedUKF;
pub use attitude_ekf::AttitudeEKF as NativeAttitudeEKF;
pub use attitude_mekf::AttitudeMEKF as NativeAttitudeMEKF;
pub use attitude_sequential_srukf::AttitudeSequentialSRUKF as NativeAttitudeSequentialSRUKF;
pub use attitude_sequential_ukf::AttitudeSequentialUKF as NativeAttitudeSequentialUKF;
pub use base::AttitudeEstimator;
pub use base::AttitudeEstimator as NativeAttitudeEstimator;
pub use base_py::PyAttitudeEstimator;

#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_batched_ukf.rs"]
mod test_attitude_batched_ukf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_ekf.rs"]
mod test_attitude_ekf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_mekf.rs"]
mod test_attitude_mekf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_sequential_srukf.rs"]
mod test_attitude_sequential_srukf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_attitude_sequential_ukf.rs"]
mod test_attitude_sequential_ukf;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_base.rs"]
mod test_base;
#[cfg(test)]
#[path = "../../../../../../tests/backends/rust/hardware/ADCS_subsystem/estimator/attitude_estimator/test_direction_correction.rs"]
mod test_direction_correction;

pub fn register(module: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    attitude_mekf_py::register(module)?;
    attitude_ekf_py::register(module)?;
    attitude_sequential_ukf_py::register(module)?;
    attitude_batched_ukf_py::register(module)?;
    attitude_sequential_srukf_py::register(module)
}

#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod bias;
mod bias_py;
pub mod fixed_bias;
mod fixed_bias_py;
pub mod linear_drift;
mod linear_drift_py;
pub mod random_walk_bias;
mod random_walk_bias_py;
pub mod scale_factor_error;
mod scale_factor_error_py;

pub use fixed_bias::FixedBias as NativeFixedBias;
pub use linear_drift::LinearDrift as NativeLinearDrift;
pub use random_walk_bias::RandomWalkBias as NativeRandomWalkBias;
pub use scale_factor_error::ScaleFactorError as NativeScaleFactorError;

#[pymodule]
fn _bias(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<bias_py::PyBias>()?;
    module.add_class::<fixed_bias_py::FixedBias>()?;
    module.add_class::<linear_drift_py::LinearDrift>()?;
    module.add_class::<random_walk_bias_py::RandomWalkBias>()?;
    module.add_class::<scale_factor_error_py::ScaleFactorError>()?;
    Ok(())
}

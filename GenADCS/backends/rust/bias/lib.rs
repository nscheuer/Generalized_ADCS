#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod bias;
mod bias_py;
mod fixed_bias;
mod linear_drift;
mod random_walk_bias;
mod scale_factor_error;

#[pymodule]
fn _bias(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<bias_py::PyBias>()?;
    module.add_class::<fixed_bias::FixedBias>()?;
    module.add_class::<linear_drift::LinearDrift>()?;
    module.add_class::<random_walk_bias::RandomWalkBias>()?;
    module.add_class::<scale_factor_error::ScaleFactorError>()?;
    Ok(())
}

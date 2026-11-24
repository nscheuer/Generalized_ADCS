#![allow(non_local_definitions)]

pub mod philox;
#[cfg(feature = "python-extension")]
mod philox_py;

#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

#[cfg(feature = "python-extension")]
#[pymodule]
fn _random(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(philox_py::philox4x32, module)?)?;
    module.add_function(wrap_pyfunction!(philox_py::standard_normals, module)?)?;
    module.add_function(wrap_pyfunction!(philox_py::validate_seed, module)?)?;
    Ok(())
}

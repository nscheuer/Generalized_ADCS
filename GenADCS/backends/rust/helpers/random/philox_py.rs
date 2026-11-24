//! PyO3 adapters for the native Philox implementation.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

use crate::philox as native;

#[pyfunction]
pub fn philox4x32(seed: u64, tick: u64, sensor: u32, block: u32) -> (u32, u32, u32, u32) {
    let words = native::philox4x32(seed, tick, sensor, block);
    (words[0], words[1], words[2], words[3])
}

#[pyfunction]
pub fn standard_normals(
    py: Python<'_>,
    seed: u64,
    tick: u64,
    sensor: u32,
    count: usize,
) -> PyResult<PyObject> {
    let values = native::standard_normals(seed, tick, sensor, count);
    let numpy = py.import("numpy")?;
    Ok(numpy.call_method1("asarray", (values,))?.into())
}

#[pyfunction]
pub fn validate_seed(seed: &PyAny) -> PyResult<u64> {
    if seed.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "seed must be an integer in [0, 2**64)",
        ));
    }
    seed.extract::<u64>().map_err(|_| {
        if seed.is_instance_of::<PyFloat>() || seed.is_instance_of::<PyInt>() {
            PyValueError::new_err("seed must be an integer in [0, 2**64)")
        } else {
            PyTypeError::new_err("seed must be an integer in [0, 2**64)")
        }
    })
}

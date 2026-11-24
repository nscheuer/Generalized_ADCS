//! PyO3 and NumPy adapters for native spherical-harmonic utilities.

use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::_spherical_harmonics;

#[pyfunction]
pub(super) fn _schmidt_table(py: Python<'_>, nmax: usize) -> PyResult<PyObject> {
    let table = _spherical_harmonics::schmidt_table(nmax);
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (table,))?
        .into())
}

#[pyfunction]
pub(super) fn read_shc(py: Python<'_>, path: &str, nmin: usize, nmax: usize) -> PyResult<PyObject> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| PyValueError::new_err(format!("could not read SHC file: {error}")))?;
    let coefficients =
        _spherical_harmonics::parse_shc(&contents, nmin, nmax).map_err(PyValueError::new_err)?;
    let side = coefficients.nmax + 1;
    let mut g = Vec::with_capacity(coefficients.epochs.len());
    let mut h = Vec::with_capacity(coefficients.epochs.len());
    for epoch in 0..coefficients.epochs.len() {
        let start = epoch * side * side;
        g.push(coefficients.g[start..start + side * side].to_vec());
        h.push(coefficients.h[start..start + side * side].to_vec());
    }
    let numpy = py.import("numpy")?;
    let epochs = numpy.call_method1("asarray", (coefficients.epochs,))?;
    let g = numpy.call_method1("asarray", (g,))?;
    let h = numpy.call_method1("asarray", (h,))?;
    Ok((epochs, g, h).into_py(py))
}

pub(super) fn vec3_from_py(py: Python<'_>, value: &PyAny) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    let vector: [f64; 3] = values
        .try_into()
        .map_err(|_| PyValueError::new_err("position must have shape (3,)"))?;
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("position must be finite"));
    }
    Ok(vector)
}

pub(super) fn py_vec3(py: Python<'_>, value: [f64; 3]) -> PyObject {
    value.to_pyarray(py).into_py(py)
}

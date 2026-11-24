#[pyclass(name = "_RotationParameter")]
struct RotationParameter;

#[pyfunction]
fn cross(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    pyvec(py, cross3(fixed(py, a, "a")?, fixed(py, b, "b")?))
}

#[pyfunction]
fn norm(py: Python<'_>, v: &PyAny) -> PyResult<f64> {
    Ok(vector_norm(&vals(py, v, "vector")?))
}

#[pyfunction]
fn normalize(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    pyvec(py, vector_normalize(&vals(py, v, "vector")?))
}

#[pyfunction]
fn unit(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    let value = vector_unit(&vals(py, v, "vector")?).map_err(PyValueError::new_err)?;
    pyvec(py, value)
}

#[pyfunction]
fn skew(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    pyvec(py, vector_skew(fixed::<3>(py, v, "vector")?))
}

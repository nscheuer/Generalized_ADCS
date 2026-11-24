#[pyfunction]
fn _normalize_quaternion(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, normalize_quaternion_native(fixed(py, q, "quaternion")?).map_err(PyValueError::new_err)?)
}
#[pyfunction]
fn _quaternion_norm(py: Python<'_>, q: &PyAny) -> PyResult<f64> {
    Ok(quaternion_norm_native(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _jpl_to_hamilton(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, jpl_to_hamilton_native(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _hamilton_to_jpl(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, hamilton_to_jpl_native(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _conjugate(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, conjugate_native(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _inverse(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, inverse_native(fixed(py, q, "quaternion")?).map_err(PyValueError::new_err)?)
}
#[pyfunction]
fn _multiply(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    pyvec(py, multiply_native(fixed(py, a, "left quaternion")?, fixed(py, b, "right quaternion")?))
}
#[pyfunction]
fn _relative_quaternion(py: Python<'_>, r: &PyAny, t: &PyAny) -> PyResult<PyObject> {
    let result = relative_quaternion_native(fixed(py, r, "reference")?, fixed(py, t, "target")?)
        .map_err(PyValueError::new_err)?;
    pyvec(py, result)
}
#[pyfunction]
fn _rotate_vector(py: Python<'_>, q: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    pyvec(py, rotate_vector_native(fixed(py, q, "quaternion")?, fixed(py, v, "vector")?))
}
#[pyfunction]
fn _quaternion_to_dcm(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &quaternion_to_dcm_native(fixed(py, q, "quaternion")?)))
}

#[pyfunction]
fn rodrigues_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    pyvec(py, rodrigues_to_quaternion_native(fixed::<3>(py, x, "parameters")?))
}

#[pyfunction]
fn quaternion_to_rodrigues(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let quaternion = fixed::<4>(py, x, "quaternion")?;
    pyvec(py, quaternion_to_rodrigues_native(quaternion).map_err(PyValueError::new_err)?)
}

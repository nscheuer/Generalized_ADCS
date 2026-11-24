#[pyfunction]
fn mrp_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    pyvec(py, mrp_to_quaternion_native(fixed::<3>(py, x, "parameters")?))
}

#[pyfunction]
fn quaternion_to_mrp(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let quaternion = fixed::<4>(py, x, "quaternion")?;
    pyvec(py, quaternion_to_mrp_native(quaternion).map_err(PyValueError::new_err)?)
}

#[pyfunction]
fn mrp_shadow_set(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let parameters = fixed::<3>(py, x, "parameters")?;
    pyvec(py, mrp_shadow_set_native(parameters).map_err(PyValueError::new_err)?)
}

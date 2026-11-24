#[pyfunction]
fn rotation_vector_to_quaternion(py: Python<'_>, r: &PyAny) -> PyResult<PyObject> {
    let vector = fixed::<3>(py, r, "rotation_vector")?;
    pyvec(py, rotation_vector_to_quaternion_native(vector).map_err(PyValueError::new_err)?)
}

#[pyfunction]
fn quaternion_to_rotation_vector(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let quaternion = fixed::<4>(py, q, "quaternion")?;
    pyvec(py, quaternion_to_rotation_vector_native(quaternion).map_err(PyValueError::new_err)?)
}

#[pyfunction]
fn slerp(py: Python<'_>, a: &PyAny, b: &PyAny, t: f64) -> PyResult<PyObject> {
    let q0 = fixed::<4>(py, a, "q0")?;
    let q1 = fixed::<4>(py, b, "q1")?;
    pyvec(py, slerp_native(q0, q1, t).map_err(PyValueError::new_err)?)
}

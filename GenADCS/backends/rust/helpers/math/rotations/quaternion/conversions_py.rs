#[pyfunction]
fn _axis_angle_to_quaternion(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<PyObject> {
    let axis = fixed::<3>(py, axis, "axis")?;
    let quaternion = axis_angle_to_quaternion_native(axis, angle)
        .map_err(PyValueError::new_err)?;
    pyvec(py, quaternion)
}

#[pyfunction]
fn _dcm_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let matrix = mat3(py, x, "matrix")?;
    let quaternion = dcm_to_quaternion_native(matrix)
        .map_err(PyValueError::new_err)?;
    pyvec(py, quaternion)
}

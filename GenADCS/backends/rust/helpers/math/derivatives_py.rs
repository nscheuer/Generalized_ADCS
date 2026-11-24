#[pyfunction]
fn quaternion_derivative(py: Python<'_>, q: &PyAny, w: &PyAny) -> PyResult<PyObject> {
    pyvec(
        py,
        quaternion_derivative_native(
            fixed::<4>(py, q, "q")?,
            fixed::<3>(py, w, "angular_velocity")?,
        ),
    )
}
#[pyfunction]
fn quaternion_rate_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, quaternion_rate_matrix_native(fixed::<4>(py, q, "q")?))
}
#[pyfunction]
fn angular_velocity_from_quaternion_derivative(
    py: Python<'_>,
    q: &PyAny,
    qd: &PyAny,
) -> PyResult<PyObject> {
    pyvec(
        py,
        angular_velocity_from_quaternion_derivative_native(
            fixed::<4>(py, q, "q")?,
            fixed::<4>(py, qd, "qdot")?,
        ),
    )
}
#[pyfunction]
fn quaternion_normalization_jacobian(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let result = quaternion_normalization_jacobian_native(fixed::<4>(py, q, "quaternion")?)
        .map_err(PyValueError::new_err)?;
    pyvec(py, result)
}
#[pyfunction]
fn rotated_vector_quaternion_jacobian(py: Python<'_>, q: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    pyvec(
        py,
        rotated_vector_quaternion_jacobian_native(
            fixed::<4>(py, q, "q")?,
            fixed::<3>(py, v, "vector")?,
        ),
    )
}
#[pyfunction]
fn quaternion_to_euler321(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, quaternion_to_euler321_native(fixed::<4>(py, q, "q")?))
}

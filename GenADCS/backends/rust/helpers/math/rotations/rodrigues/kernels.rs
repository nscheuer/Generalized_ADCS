#[pyfunction]
fn rodrigues_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let p = fixed::<3>(py, x, "parameters")?;
    let n = (1.0 + dot(&p, &p)).sqrt();
    pyvec(py, [1.0 / n, p[0] / n, p[1] / n, p[2] / n])
}
#[pyfunction]
fn quaternion_to_rodrigues(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let mut q = qnormalize(fixed::<4>(py, x, "quaternion")?)?;
    if q[0] < 0.0 {
        q = q.map(|v| -v)
    }
    if q[0].abs() <= 1e-12 {
        return Err(PyValueError::new_err(
            "Rodrigues parameters are singular at a 180-degree rotation",
        ));
    }
    pyvec(py, [q[1] / q[0], q[2] / q[0], q[3] / q[0]])
}

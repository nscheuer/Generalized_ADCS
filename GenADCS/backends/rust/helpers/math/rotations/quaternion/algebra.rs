#[pyfunction]
fn _normalize_quaternion(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, qnormalize(fixed(py, q, "quaternion")?)?)
}
#[pyfunction]
fn _quaternion_norm(py: Python<'_>, q: &PyAny) -> PyResult<f64> {
    Ok(qnorm(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _jpl_to_hamilton(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let [x, y, z, w] = fixed::<4>(py, q, "quaternion")?;
    pyvec(py, [w, -x, -y, -z])
}
#[pyfunction]
fn _hamilton_to_jpl(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let [w, x, y, z] = fixed::<4>(py, q, "quaternion")?;
    pyvec(py, [-x, -y, -z, w])
}
#[pyfunction]
fn _conjugate(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, qconj(fixed(py, q, "quaternion")?))
}
#[pyfunction]
fn _inverse(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    pyvec(py, qinv(fixed(py, q, "quaternion")?)?)
}
#[pyfunction]
fn _multiply(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    pyvec(
        py,
        qmul(
            fixed(py, a, "left quaternion")?,
            fixed(py, b, "right quaternion")?,
        ),
    )
}
#[pyfunction]
fn _relative_quaternion(py: Python<'_>, r: &PyAny, t: &PyAny) -> PyResult<PyObject> {
    let q = qnormalize(qmul(
        fixed(py, t, "target")?,
        qinv(fixed(py, r, "reference")?)?,
    ))?;
    pyvec(py, if q[0] < 0.0 { q.map(|v| -v) } else { q })
}
#[pyfunction]
fn _rotate_vector(py: Python<'_>, q: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    pyvec(
        py,
        qrotate(fixed(py, q, "quaternion")?, fixed(py, v, "vector")?),
    )
}
#[pyfunction]
fn _quaternion_to_dcm(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &qdcm(fixed(py, q, "quaternion")?)))
}

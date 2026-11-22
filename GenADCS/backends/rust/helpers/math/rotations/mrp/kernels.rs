#[pyfunction]
fn mrp_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let p = fixed::<3>(py, x, "parameters")?;
    let s = dot(&p, &p);
    let d = 1.0 + s;
    pyvec(
        py,
        [
            (1.0 - s) / d,
            2.0 * p[0] / d,
            2.0 * p[1] / d,
            2.0 * p[2] / d,
        ],
    )
}
#[pyfunction]
fn quaternion_to_mrp(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let mut q = qnormalize(fixed::<4>(py, x, "quaternion")?)?;
    if q[0] < 0.0 {
        q = q.map(|v| -v)
    }
    let mut p = [
        q[1] / (1.0 + q[0]),
        q[2] / (1.0 + q[0]),
        q[3] / (1.0 + q[0]),
    ];
    let s = dot(&p, &p);
    if s > 1.0 {
        p = p.map(|v| -v / s)
    }
    pyvec(py, p)
}
#[pyfunction]
fn mrp_shadow_set(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let p = fixed::<3>(py, x, "parameters")?;
    let n = dot(&p, &p);
    if n == 0.0 {
        return Err(PyValueError::new_err(
            "the identity MRP has no finite shadow set",
        ));
    }
    pyvec(py, p.map(|v| -v / n))
}

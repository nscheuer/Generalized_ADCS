#[pyfunction]
fn rotation_vector_to_quaternion(py: Python<'_>, r: &PyAny) -> PyResult<PyObject> {
    let v = fixed::<3>(py, r, "rotation_vector")?;
    let a = dot(&v, &v).sqrt();
    if a < 1e-12 {
        let q = [1.0 - a * a / 8.0, v[0] * 0.5, v[1] * 0.5, v[2] * 0.5];
        pyvec(py, qnormalize(q)?)
    } else {
        let s = (a / 2.0).sin() / a;
        pyvec(py, [(a / 2.0).cos(), v[0] * s, v[1] * s, v[2] * s])
    }
}
#[pyfunction]
fn quaternion_to_rotation_vector(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let mut q = qnormalize(fixed::<4>(py, q, "quaternion")?)?;
    if q[0] < 0.0 {
        q = q.map(|v| -v)
    }
    let s = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let v = if s < 1e-12 {
        [2.0 * q[1], 2.0 * q[2], 2.0 * q[3]]
    } else {
        let a = 2.0 * s.atan2(q[0]) / s;
        [q[1] * a, q[2] * a, q[3] * a]
    };
    pyvec(py, v)
}
#[pyfunction]
fn slerp(py: Python<'_>, a: &PyAny, b: &PyAny, t: f64) -> PyResult<PyObject> {
    let a = qnormalize(fixed::<4>(py, a, "q0")?)?;
    let mut b = qnormalize(fixed::<4>(py, b, "q1")?)?;
    let mut d = dot(&a, &b);
    if d < 0.0 {
        b = b.map(|v| -v);
        d = -d
    }
    let q = if d > 0.9995 {
        qnormalize(std::array::from_fn(|i| a[i] + t * (b[i] - a[i])))?
    } else {
        let th = d.clamp(-1.0, 1.0).acos();
        let den = th.sin();
        let x = ((1.0 - t) * th).sin() / den;
        let y = (t * th).sin() / den;
        std::array::from_fn(|i| x * a[i] + y * b[i])
    };
    pyvec(py, q)
}

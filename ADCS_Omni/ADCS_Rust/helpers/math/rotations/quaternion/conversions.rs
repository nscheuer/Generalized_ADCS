#[pyfunction]
fn _axis_angle_to_quaternion(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<PyObject> {
    let a = fixed::<3>(py, axis, "axis")?;
    let n = dot(&a, &a).sqrt();
    if n == 0.0 {
        return Err(PyValueError::new_err("axis must be nonzero"));
    }
    let s = (angle / 2.0).sin() / n;
    pyvec(py, [(angle / 2.0).cos(), a[0] * s, a[1] * s, a[2] * s])
}
#[pyfunction]
fn _dcm_to_quaternion(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let m = mat3(py, x, "matrix")?;
    let tr = m.trace();
    let q = if tr > 0.0 {
        let s = 2.0 * (tr + 1.0).sqrt();
        [
            0.25 * s,
            (m[(2, 1)] - m[(1, 2)]) / s,
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(1, 0)] - m[(0, 1)]) / s,
        ]
    } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
        let s = 2.0 * (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt();
        [
            (m[(2, 1)] - m[(1, 2)]) / s,
            0.25 * s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
        ]
    } else if m[(1, 1)] > m[(2, 2)] {
        let s = 2.0 * (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt();
        [
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            0.25 * s,
            (m[(1, 2)] + m[(2, 1)]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt();
        [
            (m[(1, 0)] - m[(0, 1)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
            (m[(1, 2)] + m[(2, 1)]) / s,
            0.25 * s,
        ]
    };
    let q = qnormalize(q)?;
    pyvec(py, if q[0] < 0.0 { q.map(|v| -v) } else { q })
}

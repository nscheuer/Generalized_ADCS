#[pyfunction]
fn quaternion_derivative(py: Python<'_>, q: &PyAny, w: &PyAny) -> PyResult<PyObject> {
    let q = fixed::<4>(py, q, "q")?;
    let w = fixed::<3>(py, w, "angular_velocity");
    let w = w?;
    pyvec(
        py,
        [
            0.5 * (q[1] * w[0] + q[2] * w[1] + q[3] * w[2]),
            -0.5 * (q[0] * w[0] + w[1] * q[3] - w[2] * q[2]),
            -0.5 * (q[0] * w[1] + w[2] * q[1] - w[0] * q[3]),
            -0.5 * (q[0] * w[2] + w[0] * q[2] - w[1] * q[1]),
        ],
    )
}
#[pyfunction]
fn quaternion_rate_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let [w, x, y, z] = fixed::<4>(py, q, "q")?;
    pyvec(
        py,
        vec![
            vec![x / 2., y / 2., z / 2.],
            vec![-w / 2., -z / 2., y / 2.],
            vec![z / 2., -w / 2., -x / 2.],
            vec![-y / 2., x / 2., -w / 2.],
        ],
    )
}
#[pyfunction]
fn angular_velocity_from_quaternion_derivative(
    py: Python<'_>,
    q: &PyAny,
    qd: &PyAny,
) -> PyResult<PyObject> {
    let [w, x, y, z] = fixed::<4>(py, q, "q")?;
    let [a, b, c, d] = fixed::<4>(py, qd, "qdot")?;
    pyvec(
        py,
        [
            2. * (x * a - w * b + z * c - y * d),
            2. * (y * a - z * b - w * c + x * d),
            2. * (z * a + y * b - x * c - w * d),
        ],
    )
}
#[pyfunction]
fn quaternion_normalization_jacobian(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let q = fixed::<4>(py, q, "quaternion")?;
    let n = qnorm(q);
    if n == 0.0 {
        return Err(PyValueError::new_err("quaternion must have nonzero norm"));
    }
    pyvec(
        py,
        (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if i == j {
                            1.0 / n - q[i] * q[j] / n.powi(3)
                        } else {
                            -q[i] * q[j] / n.powi(3)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
}
#[pyfunction]
fn rotated_vector_quaternion_jacobian(py: Python<'_>, q: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    let q = fixed::<4>(py, q, "q")?;
    let v = fixed::<3>(py, v, "vector")?;
    let u = [q[1], q[2], q[3]];
    let c = cross3(u, v);
    let mut out = vec![
        vec![2. * c[0], 0., 0., 0.],
        vec![2. * c[1], 0., 0., 0.],
        vec![2. * c[2], 0., 0., 0.],
    ];
    let dotuv = dot(&u, &v);
    let sv = vec![
        vec![0., -v[2], v[1]],
        vec![v[2], 0., -v[0]],
        vec![-v[1], v[0], 0.],
    ];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j + 1] = 2.
                * (if i == j { dotuv } else { 0. } + u[i] * v[j]
                    - 2. * v[i] * u[j]
                    - q[0] * sv[i][j]);
        }
    }
    pyvec(py, out)
}
#[pyfunction]
fn quaternion_to_euler321(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let [w, x, y, z] = fixed::<4>(py, q, "q")?;
    pyvec(
        py,
        [
            f64::atan2(2. * (w * x + y * z), 1. - 2. * (x * x + y * y)),
            (2. * (w * y - z * x)).clamp(-1., 1.).asin(),
            f64::atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z)),
        ],
    )
}

fn vals(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<Vec<f64>> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("ndim")?.extract::<usize>()? != 1 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite 1D array"
        )));
    }
    let v: Vec<f64> = a.call_method0("tolist")?.extract()?;
    if v.iter().any(|n| !n.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(v)
}
fn fixed<const N: usize>(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[f64; N]> {
    vals(py, x, name)?
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a finite {N}-vector")))
}
fn mat3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<M> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("shape")?.extract::<(usize, usize)>().ok() != Some((3, 3)) {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite 3x3 array"
        )));
    }
    let rows: Vec<Vec<f64>> = a.call_method0("tolist")?.extract()?;
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(M::from_row_slice(&flat))
}
fn pyvec(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
fn pymat(py: Python<'_>, matrix: &M) -> PyObject {
    matrix.to_pyarray(py).into_py(py)
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    (0..N).map(|i| a[i] * b[i]).sum()
}
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
    ]
}
fn qnorm(q: [f64; 4]) -> f64 {
    dot(&q, &q).sqrt()
}
fn qnormalize(q: [f64; 4]) -> PyResult<[f64; 4]> {
    let n = qnorm(q);
    if n == 0.0 {
        return Err(PyValueError::new_err("quaternion must have nonzero norm"));
    }
    Ok(q.map(|v| v / n))
}
fn qconj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn qinv(q: [f64; 4]) -> PyResult<[f64; 4]> {
    let n = dot(&q, &q);
    if n == 0.0 {
        return Err(PyValueError::new_err("zero quaternion has no inverse"));
    }
    Ok(qconj(q).map(|v| v / n))
}
fn qrotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let c = cross3(u, v);
    let cc = cross3(u, c);
    [
        v[0] + 2.0 * (q[0] * c[0] + cc[0]),
        v[1] + 2.0 * (q[0] * c[1] + cc[1]),
        v[2] + 2.0 * (q[0] * c[2] + cc[2]),
    ]
}
fn qdcm(q: [f64; 4]) -> M {
    let [w, x, y, z] = q;
    M::from_row_slice(&[
        w * w + x * x - y * y - z * z,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    ])
}
fn mm(a: M, b: M) -> M {
    a * b
}
fn tr(a: M) -> M {
    a.transpose()
}
fn mv(a: M, b: [f64; 3]) -> [f64; 3] {
    let v = a * SVector::<f64, 3>::from_row_slice(&b);
    [v[0], v[1], v[2]]
}
fn quatarg(py: Python<'_>, x: &PyAny) -> PyResult<[f64; 4]> {
    if x.hasattr("hamilton_components")? {
        fixed(py, x.getattr("hamilton_components")?, "quaternion")
    } else {
        fixed(py, x, "quaternion")
    }
}
fn proper_dcm(m: M) -> bool {
    (m.transpose() * m - M::identity()).norm() <= 1e-8 && (m.determinant() - 1.0).abs() <= 1e-8
}

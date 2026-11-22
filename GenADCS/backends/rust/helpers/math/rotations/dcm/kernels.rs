#[pyfunction]
fn axis_angle_to_dcm(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<PyObject> {
    let q = _axis_angle_to_quaternion(py, axis, angle)?;
    _quaternion_to_dcm(py, q.as_ref(py))
}
#[pyfunction]
fn rotate_vector(py: Python<'_>, m: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    let m = mat3(py, m, "dcm")?;
    let v = fixed::<3>(py, v, "vector")?;
    pyvec(py, mv(m, v))
}
#[pyfunction]
fn inverse_dcm(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let m = mat3(py, x, "dcm")?;
    Ok(pymat(py, &tr(m)))
}
#[pyfunction]
fn compose_dcm(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &mm(mat3(py, a, "left")?, mat3(py, b, "right")?)))
}
#[pyfunction]
fn relative_dcm(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    let a = mat3(py, a, "target")?;
    let b = mat3(py, b, "reference")?;
    Ok(pymat(py, &mm(a, tr(b))))
}
#[pyfunction]
fn dcm_orthogonality_error(py: Python<'_>, x: &PyAny) -> PyResult<f64> {
    let m = mat3(py, x, "matrix")?;
    Ok((m.transpose() * m - M::identity()).norm())
}
#[pyfunction]
fn normalize_dcm(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let m = mat3(py, x, "matrix")?;
    let mut a = m.column(0).into_owned();
    let n = a.norm();
    if n == 0.0 {
        return Err(PyValueError::new_err("DCM has a degenerate first column"));
    }
    a /= n;
    let b0 = m.column(1).into_owned();
    let mut b = b0 - a.dot(&b0) * a;
    let n = b.norm();
    if n == 0.0 {
        return Err(PyValueError::new_err(
            "DCM has degenerate first and second columns",
        ));
    }
    b /= n;
    let c = a.cross(&b);
    let out = M::from_columns(&[a, b, c]);
    Ok(pymat(py, &out))
}

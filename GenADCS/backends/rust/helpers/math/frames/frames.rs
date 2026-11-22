#[pyfunction]
fn inertial_to_body_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let q: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (q,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if q.len() != 4 || q.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("q_b_from_i must be finite 4-vector"));
    }
    Ok(pymat(py, &qmat(q.try_into().unwrap())))
}
#[pyfunction]
fn body_to_inertial_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    let m = inertial_to_body_matrix(py, q)?;
    let x = matrix(py, m.as_ref(py), "matrix")?;
    Ok(pymat(py, &tr(x)))
}
#[pyfunction]
fn inertial_to_body(py: Python<'_>, v: &PyAny, q: &PyAny) -> PyResult<PyObject> {
    let m = inertial_to_body_matrix(py, q)?;
    pyarr(
        py,
        mv(
            matrix(py, m.as_ref(py), "matrix")?,
            vec3(py, v, "vector_i")?,
        ),
    )
}
#[pyfunction]
fn body_to_inertial(py: Python<'_>, v: &PyAny, q: &PyAny) -> PyResult<PyObject> {
    let m = body_to_inertial_matrix(py, q)?;
    pyarr(
        py,
        mv(
            matrix(py, m.as_ref(py), "matrix")?,
            vec3(py, v, "vector_b")?,
        ),
    )
}
#[pyfunction]
fn transform_vector(py: Python<'_>, v: &PyAny, m: &PyAny) -> PyResult<PyObject> {
    pyarr(
        py,
        mv(
            matrix(py, m, "source_to_target_dcm")?,
            vec3(py, v, "vector")?,
        ),
    )
}
#[pyfunction]
fn transform_covariance(py: Python<'_>, c: &PyAny, m: &PyAny) -> PyResult<PyObject> {
    let m = matrix(py, m, "source_to_target_dcm")?;
    Ok(pymat(
        py,
        &mm(mm(m, matrix(py, c, "covariance_source")?), tr(m)),
    ))
}
#[pyfunction]
fn gcrf_to_lvlh_matrix(py: Python<'_>, p: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(
        py,
        &lvlh(vec3(py, p, "position_gcrf")?, vec3(py, v, "velocity_gcrf")?)?,
    ))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,jd_tt,xp=0.0,yp=0.0))]
fn gcrf_to_ecef_matrix(
    py: Python<'_>,
    jd_ut1: f64,
    jd_tt: f64,
    xp: f64,
    yp: f64,
) -> PyResult<PyObject> {
    Ok(pymat(py, &mm(teme_ecef(jd_ut1, xp, yp), gcrf_teme(jd_tt))))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,jd_tt,xp=0.0,yp=0.0))]
fn ecef_to_gcrf_matrix(
    py: Python<'_>,
    jd_ut1: f64,
    jd_tt: f64,
    xp: f64,
    yp: f64,
) -> PyResult<PyObject> {
    let m = mm(teme_ecef(jd_ut1, xp, yp), gcrf_teme(jd_tt));
    Ok(pymat(py, &tr(m)))
}
#[pyfunction]
fn gcrf_to_teme_matrix(py: Python<'_>, jd_tt: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &gcrf_teme(jd_tt)))
}
#[pyfunction]
fn teme_to_gcrf_matrix(py: Python<'_>, jd_tt: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &tr(gcrf_teme(jd_tt))))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,xp=0.0,yp=0.0))]
fn teme_to_ecef_matrix(py: Python<'_>, jd_ut1: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &teme_ecef(jd_ut1, xp, yp)))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,xp=0.0,yp=0.0))]
fn ecef_to_teme_matrix(py: Python<'_>, jd_ut1: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &tr(teme_ecef(jd_ut1, xp, yp))))
}

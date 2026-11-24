fn quaternion4(py: Python<'_>, q: &PyAny) -> PyResult<[f64; 4]> {
    let values: Vec<f64> = py.import("numpy")?.call_method1("asarray", (q,))?
        .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("q_b_from_i must be finite 4-vector"));
    }
    Ok(values.try_into().unwrap())
}
#[pyfunction]
fn inertial_to_body_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &inertial_to_body_matrix_native(quaternion4(py, q)?)))
}
#[pyfunction]
fn body_to_inertial_matrix(py: Python<'_>, q: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &body_to_inertial_matrix_native(quaternion4(py, q)?)))
}
#[pyfunction]
fn inertial_to_body(py: Python<'_>, v: &PyAny, q: &PyAny) -> PyResult<PyObject> {
    pyarr(py, inertial_to_body_native(vec3(py, v, "vector_i")?, quaternion4(py, q)?))
}
#[pyfunction]
fn body_to_inertial(py: Python<'_>, v: &PyAny, q: &PyAny) -> PyResult<PyObject> {
    pyarr(py, body_to_inertial_native(vec3(py, v, "vector_b")?, quaternion4(py, q)?))
}
#[pyfunction]
fn transform_vector(py: Python<'_>, v: &PyAny, m: &PyAny) -> PyResult<PyObject> {
    pyarr(py, transform_vector_native(vec3(py, v, "vector")?, matrix(py, m, "source_to_target_dcm")?))
}
#[pyfunction]
fn transform_covariance(py: Python<'_>, c: &PyAny, m: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &transform_covariance_native(matrix(py, c, "covariance_source")?, matrix(py, m, "source_to_target_dcm")?)))
}
#[pyfunction]
fn gcrf_to_lvlh_matrix(py: Python<'_>, p: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    let result = gcrf_to_lvlh_matrix_native(vec3(py, p, "position_gcrf")?, vec3(py, v, "velocity_gcrf")?)
        .map_err(PyValueError::new_err)?;
    Ok(pymat(py, &result))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,jd_tt,xp=0.0,yp=0.0))]
fn gcrf_to_ecef_matrix(py: Python<'_>, jd_ut1: f64, jd_tt: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &gcrf_to_ecef_matrix_native(jd_ut1, jd_tt, xp, yp)))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,jd_tt,xp=0.0,yp=0.0))]
fn ecef_to_gcrf_matrix(py: Python<'_>, jd_ut1: f64, jd_tt: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &ecef_to_gcrf_matrix_native(jd_ut1, jd_tt, xp, yp)))
}
#[pyfunction]
fn gcrf_to_teme_matrix(py: Python<'_>, jd_tt: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &gcrf_to_teme_matrix_native(jd_tt)))
}
#[pyfunction]
fn teme_to_gcrf_matrix(py: Python<'_>, jd_tt: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &teme_to_gcrf_matrix_native(jd_tt)))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,xp=0.0,yp=0.0))]
fn teme_to_ecef_matrix(py: Python<'_>, jd_ut1: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &teme_to_ecef_matrix_native(jd_ut1, xp, yp)))
}
#[pyfunction]
#[pyo3(signature=(jd_ut1,xp=0.0,yp=0.0))]
fn ecef_to_teme_matrix(py: Python<'_>, jd_ut1: f64, xp: f64, yp: f64) -> PyResult<PyObject> {
    Ok(pymat(py, &ecef_to_teme_matrix_native(jd_ut1, xp, yp)))
}

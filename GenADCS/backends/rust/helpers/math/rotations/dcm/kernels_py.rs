#[pyfunction]
fn axis_angle_to_dcm(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<PyObject> {
    let axis = fixed::<3>(py, axis, "axis")?;
    let matrix = axis_angle_to_dcm_native(axis, angle).map_err(PyValueError::new_err)?;
    Ok(pymat(py, &matrix))
}

#[pyfunction]
fn rotate_vector(py: Python<'_>, m: &PyAny, v: &PyAny) -> PyResult<PyObject> {
    let matrix = mat3(py, m, "dcm")?;
    let vector = fixed::<3>(py, v, "vector")?;
    pyvec(py, rotate_vector_dcm_native(matrix, vector))
}

#[pyfunction]
fn inverse_dcm(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    Ok(pymat(py, &inverse_dcm_native(mat3(py, x, "dcm")?)))
}

#[pyfunction]
fn compose_dcm(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    let left = mat3(py, a, "left")?;
    let right = mat3(py, b, "right")?;
    Ok(pymat(py, &compose_dcm_native(left, right)))
}

#[pyfunction]
fn relative_dcm(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    let target = mat3(py, a, "target")?;
    let reference = mat3(py, b, "reference")?;
    Ok(pymat(py, &relative_dcm_native(target, reference)))
}

#[pyfunction]
fn dcm_orthogonality_error(py: Python<'_>, x: &PyAny) -> PyResult<f64> {
    Ok(dcm_orthogonality_error_native(mat3(py, x, "matrix")?))
}

#[pyfunction]
fn normalize_dcm(py: Python<'_>, x: &PyAny) -> PyResult<PyObject> {
    let normalized = normalize_dcm_native(mat3(py, x, "matrix")?)
        .map_err(PyValueError::new_err)?;
    Ok(pymat(py, &normalized))
}

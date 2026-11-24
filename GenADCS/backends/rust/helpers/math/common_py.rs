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
fn qnormalize(q: [f64; 4]) -> PyResult<[f64; 4]> {
    qnormalize_native(q).map_err(PyValueError::new_err)
}
fn qinv(q: [f64; 4]) -> PyResult<[f64; 4]> {
    qinv_native(q).map_err(PyValueError::new_err)
}
fn quatarg(py: Python<'_>, x: &PyAny) -> PyResult<[f64; 4]> {
    if x.hasattr("hamilton_components")? {
        fixed(py, x.getattr("hamilton_components")?, "quaternion")
    } else {
        fixed(py, x, "quaternion")
    }
}

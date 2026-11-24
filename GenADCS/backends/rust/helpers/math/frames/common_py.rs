fn vec3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<V> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("shape")?.extract::<(usize,)>().ok() != Some((3,)) {
        return Err(PyValueError::new_err(format!(
            "{name} has an invalid shape"
        )));
    }
    let v: V = a
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?
        .try_into()
        .unwrap();
    if v.iter().any(|n| !n.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(v)
}
fn matrix(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<M> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("shape")?.extract::<(usize, usize)>().ok() != Some((3, 3)) {
        return Err(PyValueError::new_err(format!(
            "{name} has an invalid shape"
        )));
    }
    let r: Vec<Vec<f64>> = a.call_method0("tolist")?.extract()?;
    let flat: Vec<f64> = r.into_iter().flatten().collect();
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(M::from_row_slice(&flat))
}
fn pyarr(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
fn pymat(py: Python<'_>, x: &M) -> PyObject {
    x.to_pyarray(py).into_py(py)
}

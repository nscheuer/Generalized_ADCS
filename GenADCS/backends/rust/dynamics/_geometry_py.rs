// Python and NumPy conversions for the Rust-native geometry API.

fn vec3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let a: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.len() != 3 || a.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite 3-vector"
        )));
    }
    Ok([a[0], a[1], a[2]])
}

fn quat(py: Python<'_>, x: &PyAny) -> PyResult<[f64; 4]> {
    let a: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.len() != 4 || a.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "q_b_from_i must be a finite 4-vector",
        ));
    }
    Ok([a[0], a[1], a[2], a[3]])
}

fn mat3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[[f64; 3]; 3]> {
    let a: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.len() != 3
        || a.iter()
            .any(|r| r.len() != 3 || r.iter().any(|v| !v.is_finite()))
    {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite 3x3 matrix"
        )));
    }
    Ok([
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ])
}

fn rows(py: Python<'_>, x: &PyAny, name: &str, cols: usize) -> PyResult<Vec<Vec<f64>>> {
    let a: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.iter()
        .any(|r| r.len() != cols || r.iter().any(|v| !v.is_finite()))
    {
        return Err(PyValueError::new_err(format!(
            "{name} must have {cols} finite columns"
        )));
    }
    Ok(a)
}

fn vals(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<Vec<f64>> {
    let a: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must be finite")));
    }
    Ok(a)
}

fn out(py: Python<'_>, v: [f64; 3]) -> PyObject {
    v.to_pyarray(py).into_py(py)
}

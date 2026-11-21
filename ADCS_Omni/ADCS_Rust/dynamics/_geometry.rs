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
fn rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            w * w + x * x - y * y - z * z,
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ]
}
fn mv(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn out(py: Python<'_>, v: [f64; 3]) -> PyObject {
    v.to_pyarray(py).into_py(py)
}

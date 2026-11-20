#[pyclass(name = "_RotationParameter")]
struct RotationParameter;

#[pyfunction]
fn cross(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    pyvec(py, cross3(fixed(py, a, "a")?, fixed(py, b, "b")?))
}
#[pyfunction]
fn norm(py: Python<'_>, v: &PyAny) -> PyResult<f64> {
    Ok(vals(py, v, "vector")?
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt())
}
#[pyfunction]
fn normalize(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    let x = vals(py, v, "vector")?;
    let n = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    pyvec(
        py,
        if n == 0.0 {
            x.iter().map(|_| 0.0).collect::<Vec<_>>()
        } else {
            x.iter().map(|a| a / n).collect()
        },
    )
}
#[pyfunction]
fn unit(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    let x = vals(py, v, "vector")?;
    let n = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    if n == 0.0 {
        return Err(PyValueError::new_err("cannot normalize a zero vector"));
    }
    pyvec(py, x.iter().map(|a| a / n).collect::<Vec<_>>())
}
#[pyfunction]
fn skew(py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
    let [x, y, z] = fixed::<3>(py, v, "vector")?;
    pyvec(py, vec![vec![0., -z, y], vec![z, 0., -x], vec![-y, x, 0.]])
}

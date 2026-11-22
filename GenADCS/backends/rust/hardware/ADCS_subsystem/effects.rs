#[pyfunction]
#[pyo3(signature = (value, effects, dt, rng=None))]
fn apply_effects(
    py: Python<'_>,
    mut value: PyObject,
    effects: Vec<PyObject>,
    dt: f64,
    rng: Option<PyObject>,
) -> PyResult<PyObject> {
    let rng = rng.unwrap_or_else(|| py.None());
    for effect in effects {
        value = effect
            .as_ref(py)
            .call_method1("apply", (value, dt, rng.clone_ref(py)))?
            .into_py(py);
    }
    Ok(value)
}

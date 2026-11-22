#[pyclass(name = "Magnetorquer")]
pub struct Magnetorquer {
    #[pyo3(get, set)]
    position: PyObject,
    #[pyo3(get, set)]
    orientation: PyObject,
    #[pyo3(get, set)]
    max_dipole: f64,
    #[pyo3(get, set)]
    misalignment: PyObject,
    effects: Vec<PyObject>,
}

#[pymethods]
impl Magnetorquer {
    #[new]
    #[pyo3(signature = (position, orientation, max_dipole, misalignment=None, effects=None))]
    fn new(
        py: Python<'_>,
        position: PyObject,
        orientation: PyObject,
        max_dipole: f64,
        misalignment: Option<PyObject>,
        effects: Option<Vec<PyObject>>,
    ) -> Self {
        Self {
            position,
            orientation,
            max_dipole,
            misalignment: misalignment.unwrap_or_else(|| (0.0, 0.0, 0.0).into_py(py)),
            effects: effects.unwrap_or_default(),
        }
    }

    #[getter]
    fn effects(&self, py: Python<'_>) -> PyObject {
        self.effects.clone().into_py(py)
    }

    #[setter]
    fn set_effects(&mut self, effects: Vec<PyObject>) {
        self.effects = effects;
    }

    #[getter]
    fn noises(&self, py: Python<'_>) -> PyResult<PyObject> {
        effects_of_type(py, &self.effects, "GenADCS.backends.python.noise", "Noise")
    }

    #[getter]
    fn biases(&self, py: Python<'_>) -> PyResult<PyObject> {
        effects_of_type(py, &self.effects, "GenADCS.backends.python.bias", "Bias")
    }

    #[pyo3(signature = (value, dt, rng=None))]
    fn apply_errors(
        &self,
        py: Python<'_>,
        value: PyObject,
        dt: f64,
        rng: Option<PyObject>,
    ) -> PyResult<PyObject> {
        apply_effects(py, value, self.effects.clone(), dt, rng)
    }
}

fn effects_of_type(
    py: Python<'_>,
    effects: &[PyObject],
    module: &str,
    type_name: &str,
) -> PyResult<PyObject> {
    let class = py.import(module)?.getattr(type_name)?;
    let selected = effects
        .iter()
        .filter_map(|effect| match effect.as_ref(py).is_instance(class) {
            Ok(true) => Some(Ok(effect.clone_ref(py))),
            Ok(false) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(selected.into_py(py))
}

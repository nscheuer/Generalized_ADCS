#[pyclass(name = "ReactionWheel")]
pub struct ReactionWheel {
    #[pyo3(get, set)]
    position: PyObject,
    #[pyo3(get, set)]
    orientation: PyObject,
    #[pyo3(get, set)]
    rotational_inertia: f64,
    #[pyo3(get, set)]
    max_torque: f64,
    #[pyo3(get, set)]
    max_momentum: f64,
    #[pyo3(get, set)]
    misalignment: PyObject,
    effects: Vec<PyObject>,
}

#[pymethods]
impl ReactionWheel {
    #[new]
    #[pyo3(signature = (position, orientation, rotational_inertia, max_torque, max_momentum, misalignment=None, effects=None))]
    fn new(
        py: Python<'_>,
        position: PyObject,
        orientation: PyObject,
        rotational_inertia: f64,
        max_torque: f64,
        max_momentum: f64,
        misalignment: Option<PyObject>,
        effects: Option<Vec<PyObject>>,
    ) -> Self {
        Self {
            position,
            orientation,
            rotational_inertia,
            max_torque,
            max_momentum,
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

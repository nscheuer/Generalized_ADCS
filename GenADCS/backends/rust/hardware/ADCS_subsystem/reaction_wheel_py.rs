#[pyclass(name = "ReactionWheel")]
pub struct ReactionWheel {
    position: PyObject,
    orientation: PyObject,
    rotational_inertia: f64,
    max_torque: f64,
    max_momentum: f64,
    misalignment: PyObject,
    effects: Vec<PyObject>,
    native: ReactionWheelData,
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
    ) -> PyResult<Self> {
        let misalignment = misalignment.unwrap_or_else(|| (0.0, 0.0, 0.0).into_py(py));
        let native = ReactionWheelData::new(
            python_vector(py, &position).unwrap_or_else(|_| vec![0.0; 3]),
            python_vector(py, &orientation).unwrap_or_else(|_| vec![0.0; 3]),
            rotational_inertia,
            max_torque,
            max_momentum,
            python_vector(py, &misalignment).unwrap_or_else(|_| vec![0.0; 3]),
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            position,
            orientation,
            rotational_inertia,
            max_torque,
            max_momentum,
            misalignment,
            effects: effects.unwrap_or_default(),
            native,
        })
    }

    #[getter]
    fn position(&self, py: Python<'_>) -> PyObject {
        self.position.clone_ref(py)
    }

    #[setter]
    fn set_position(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        if let Ok(v) = python_vector(py, &value) { self.native.position = v; }
        self.position = value;
        Ok(())
    }

    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyObject {
        self.orientation.clone_ref(py)
    }

    #[setter]
    fn set_orientation(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        if let Ok(v) = python_vector(py, &value) { self.native.orientation = v; }
        self.orientation = value;
        Ok(())
    }

    #[getter]
    fn rotational_inertia(&self) -> f64 {
        self.rotational_inertia
    }

    #[setter]
    fn set_rotational_inertia(&mut self, value: f64) -> PyResult<()> {
        validate_wheel_scalar(value)?;
        self.rotational_inertia = value;
        self.native.rotational_inertia = value;
        Ok(())
    }

    #[getter]
    fn max_torque(&self) -> f64 {
        self.max_torque
    }

    #[setter]
    fn set_max_torque(&mut self, value: f64) -> PyResult<()> {
        validate_wheel_scalar(value)?;
        self.max_torque = value;
        self.native.max_torque = value;
        Ok(())
    }

    #[getter]
    fn max_momentum(&self) -> f64 {
        self.max_momentum
    }

    #[setter]
    fn set_max_momentum(&mut self, value: f64) -> PyResult<()> {
        validate_wheel_scalar(value)?;
        self.max_momentum = value;
        self.native.max_momentum = value;
        Ok(())
    }

    #[getter]
    fn misalignment(&self, py: Python<'_>) -> PyObject {
        self.misalignment.clone_ref(py)
    }

    #[setter]
    fn set_misalignment(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        if let Ok(v) = python_vector(py, &value) { self.native.misalignment = v; }
        self.misalignment = value;
        Ok(())
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

fn validate_wheel_scalar(value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "reaction wheel values must be finite",
        ))
    }
}

fn python_vector(py: Python<'_>, value: &PyObject) -> PyResult<Vec<f64>> {
    if value.as_ref(py).is_none() {
        return Ok(Vec::new());
    }
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "reaction wheel values must be finite",
        ));
    }
    Ok(values)
}

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
    noise: f64,
    bias: f64,
    #[pyo3(get, set)]
    name: Option<String>,
}

#[pymethods]
impl Magnetorquer {
    #[new]
    #[pyo3(signature = (position, orientation, max_dipole, misalignment=None, noise=0.0, bias=0.0, name=None))]
    fn new(
        py: Python<'_>,
        position: PyObject,
        orientation: PyObject,
        max_dipole: f64,
        misalignment: Option<PyObject>,
        noise: f64,
        bias: f64,
        name: Option<String>,
    ) -> PyResult<Self> {
        if !noise.is_finite() || noise < 0.0 {
            return Err(PyValueError::new_err("noise must be finite and nonnegative"));
        }
        if !bias.is_finite() {
            return Err(PyValueError::new_err("bias must be finite"));
        }
        Ok(Self {
            position,
            orientation,
            max_dipole,
            misalignment: misalignment.unwrap_or_else(|| (0.0, 0.0, 0.0).into_py(py)),
            noise,
            bias,
            name,
        })
    }

    #[getter]
    fn noise(&self) -> f64 {
        self.noise
    }

    #[setter]
    fn set_noise(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || value < 0.0 {
            return Err(PyValueError::new_err("noise must be finite and nonnegative"));
        }
        self.noise = value;
        Ok(())
    }

    #[getter]
    fn bias(&self) -> f64 {
        self.bias
    }

    #[setter]
    fn set_bias(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(PyValueError::new_err("bias must be finite"));
        }
        self.bias = value;
        Ok(())
    }
}

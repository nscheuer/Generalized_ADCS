use super::*;

#[pyclass(extends = MagneticFieldModel)]
pub(super) struct CHAOS8 {
    model: Chaos8,
}

#[pymethods]
impl CHAOS8 {
    #[new]
    #[pyo3(signature = (include_static = true))]
    fn new(include_static: bool) -> PyResult<(Self, MagneticFieldModel)> {
        let model = Chaos8::new(include_static).map_err(PyValueError::new_err)?;
        Ok((Self { model }, MagneticFieldModel))
    }

    #[getter]
    fn include_static(&self) -> bool {
        self.model.includes_static()
    }

    #[setter]
    fn set_include_static(&self, _value: bool) -> PyResult<()> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "include_static is read-only after construction",
        ))
    }

    fn field_inertial_tesla(
        &self,
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
    ) -> PyResult<PyObject> {
        let position = vec3_from_py(py, position_inertial_m)?;
        let field = self
            .model
            .field_inertial_tesla(position, epoch_jd)
            .map_err(PyValueError::new_err)?;
        Ok(py_vec3(py, field))
    }

    fn field(
        &self,
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
    ) -> PyResult<PyObject> {
        self.field_inertial_tesla(py, position_inertial_m, epoch_jd)
    }
}

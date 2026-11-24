use super::*;

#[pyclass(extends = MagneticFieldModel)]
pub(super) struct IGRF14 {
    model: Igrf14,
}

#[pymethods]
impl IGRF14 {
    #[new]
    fn new() -> PyResult<(Self, MagneticFieldModel)> {
        let model = Igrf14::new().map_err(PyValueError::new_err)?;
        Ok((Self { model }, MagneticFieldModel))
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

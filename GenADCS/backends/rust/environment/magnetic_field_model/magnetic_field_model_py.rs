use super::*;

#[pyclass(subclass)]
pub(super) struct MagneticFieldModel;

#[pymethods]
impl MagneticFieldModel {
    #[new]
    fn new() -> Self {
        Self
    }

    fn field_inertial_tesla(&self, _position_inertial_m: &PyAny, _epoch_jd: f64) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Choose a magnetic field model",
        ))
    }

    fn field(&self, _position_inertial_m: &PyAny, _epoch_jd: f64) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Choose a magnetic field model",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (position_inertial_m, epoch_jd, g, h, nmax))]
pub(super) fn magnetic_field_inertial_tesla(
    py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
    g: &PyAny,
    h: &PyAny,
    nmax: usize,
) -> PyResult<PyObject> {
    let position = vec3_from_py(py, position_inertial_m)?;
    let g = coefficients_from_py(py, g)?;
    let h = coefficients_from_py(py, h)?;
    let field = super::magnetic_field_model::magnetic_field_inertial_tesla(
        position, epoch_jd, &g, &h, nmax,
    )
    .map_err(PyValueError::new_err)?;
    Ok(py_vec3(py, field))
}

fn coefficients_from_py(py: Python<'_>, value: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    let numpy_array = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?;
    if let Ok(epochs) = numpy_array
        .call_method0("tolist")?
        .extract::<Vec<Vec<Vec<f64>>>>()
    {
        return epochs
            .into_iter()
            .next()
            .ok_or_else(|| PyValueError::new_err("coefficient arrays are empty"));
    }
    numpy_array
        .call_method0("tolist")?
        .extract::<Vec<Vec<f64>>>()
        .map_err(Into::into)
}

#[pyclass(extends = MagneticFieldModel)]
pub(super) struct WMM;

#[pymethods]
impl WMM {
    #[new]
    fn new() -> (Self, MagneticFieldModel) {
        (Self, MagneticFieldModel)
    }

    fn field_inertial_tesla(&self, _position: &PyAny, _epoch_jd: f64) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Choose a magnetic field model",
        ))
    }

    fn field(&self, _position: &PyAny, _epoch_jd: f64) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Choose a magnetic field model",
        ))
    }
}

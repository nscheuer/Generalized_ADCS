use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::satellite_panel::SatellitePanel as NativeSatellitePanel;

fn fixed_values<const N: usize>(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<[f64; N]> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.len() != N || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain {N} finite values"
        )));
    }
    Ok(std::array::from_fn(|index| values[index]))
}

#[pyclass(name = "SatellitePanel")]
#[derive(Clone)]
pub struct SatellitePanel {
    pub(super) inner: NativeSatellitePanel,
}

#[pymethods]
impl SatellitePanel {
    #[new]
    #[pyo3(signature = (position, orientation, dimensions, eta_s=0.0, eta_d=0.0, eta_a=1.0, cd=2.2))]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        dimensions: &PyAny,
        eta_s: f64,
        eta_d: f64,
        eta_a: f64,
        cd: f64,
    ) -> PyResult<Self> {
        let mut inner = NativeSatellitePanel::new(
            fixed_values(py, position, "position")?,
            fixed_values(py, orientation, "orientation")?,
            fixed_values(py, dimensions, "dimensions")?,
        );
        inner.eta_s = eta_s;
        inner.eta_d = eta_d;
        inner.eta_a = eta_a;
        inner.cd = cd;
        Ok(Self { inner })
    }

    #[getter]
    fn position(&self) -> [f64; 3] {
        self.inner.position
    }
    #[setter]
    fn set_position(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.position = fixed_values(py, value, "position")?;
        Ok(())
    }
    #[getter]
    fn orientation(&self) -> [f64; 3] {
        self.inner.orientation
    }
    #[setter]
    fn set_orientation(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.orientation = fixed_values(py, value, "orientation")?;
        Ok(())
    }
    #[getter]
    fn dimensions(&self) -> [f64; 2] {
        self.inner.dimensions
    }
    #[setter]
    fn set_dimensions(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.dimensions = fixed_values(py, value, "dimensions")?;
        Ok(())
    }
    #[getter]
    fn eta_s(&self) -> f64 {
        self.inner.eta_s
    }
    #[setter]
    fn set_eta_s(&mut self, value: f64) {
        self.inner.eta_s = value;
    }
    #[getter]
    fn eta_d(&self) -> f64 {
        self.inner.eta_d
    }
    #[setter]
    fn set_eta_d(&mut self, value: f64) {
        self.inner.eta_d = value;
    }
    #[getter]
    fn eta_a(&self) -> f64 {
        self.inner.eta_a
    }
    #[setter]
    fn set_eta_a(&mut self, value: f64) {
        self.inner.eta_a = value;
    }
    #[getter]
    fn cd(&self) -> f64 {
        self.inner.cd
    }
    #[setter]
    fn set_cd(&mut self, value: f64) {
        self.inner.cd = value;
    }
}

impl SatellitePanel {
    pub fn native(&self) -> &NativeSatellitePanel {
        &self.inner
    }
}

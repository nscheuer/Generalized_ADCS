use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::satellite_geometry::SatelliteGeometry as NativeSatelliteGeometry;
use super::satellite_panel_py::SatellitePanel;

fn vector3(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite 3-vector"
        )));
    }
    Ok([values[0], values[1], values[2]])
}

fn matrix3(py: Python<'_>, value: &PyAny) -> PyResult<[[f64; 3]; 3]> {
    let values: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.len() != 3
        || values
            .iter()
            .any(|row| row.len() != 3 || row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err("inertia must be a finite 3x3 matrix"));
    }
    Ok([
        [values[0][0], values[0][1], values[0][2]],
        [values[1][0], values[1][1], values[1][2]],
        [values[2][0], values[2][1], values[2][2]],
    ])
}

#[pyclass(name = "SatelliteGeometry")]
pub struct SatelliteGeometry {
    inner: NativeSatelliteGeometry,
}

#[pymethods]
impl SatelliteGeometry {
    #[new]
    #[pyo3(signature = (mass, inertia, panels=None, center_of_mass=None))]
    fn new(
        py: Python<'_>,
        mass: f64,
        inertia: &PyAny,
        panels: Option<Vec<Py<SatellitePanel>>>,
        center_of_mass: Option<&PyAny>,
    ) -> PyResult<Self> {
        let mut inner = NativeSatelliteGeometry::new(mass, matrix3(py, inertia)?);
        if let Some(center) = center_of_mass {
            inner.center_of_mass = vector3(py, center, "center_of_mass")?;
        }
        if let Some(panels) = panels {
            Python::with_gil(|py| {
                inner.panels = panels
                    .iter()
                    .map(|panel| panel.borrow(py).native().clone())
                    .collect();
            });
        }
        Ok(Self { inner })
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass
    }
    #[setter]
    fn set_mass(&mut self, value: f64) {
        self.inner.mass = value;
    }
    #[getter]
    fn inertia(&self) -> [[f64; 3]; 3] {
        self.inner.inertia
    }
    #[setter]
    fn set_inertia(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.inertia = matrix3(py, value)?;
        Ok(())
    }
    #[getter]
    fn center_of_mass(&self) -> [f64; 3] {
        self.inner.center_of_mass
    }
    #[setter]
    fn set_center_of_mass(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.center_of_mass = vector3(py, value, "center_of_mass")?;
        Ok(())
    }
    #[getter]
    fn panels(&self, py: Python<'_>) -> PyResult<Vec<Py<SatellitePanel>>> {
        self.inner
            .panels
            .iter()
            .cloned()
            .map(|panel| Py::new(py, SatellitePanel { inner: panel }))
            .collect()
    }
    #[setter]
    fn set_panels(&mut self, panels: Vec<Py<SatellitePanel>>) -> PyResult<()> {
        Python::with_gil(|py| {
            self.inner.panels = panels
                .iter()
                .map(|panel| panel.borrow(py).native().clone())
                .collect();
        });
        Ok(())
    }
}

impl SatelliteGeometry {
    pub fn native(&self) -> &NativeSatelliteGeometry {
        &self.inner
    }
}

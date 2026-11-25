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
    inertia_value: PyObject,
    center_value: PyObject,
    panel_values: Vec<PyObject>,
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
        let mut inner =
            NativeSatelliteGeometry::new(mass, matrix3(py, inertia).unwrap_or([[0.0; 3]; 3]));
        let center_value = center_of_mass
            .map(|v| v.into_py(py))
            .unwrap_or_else(|| (0.0, 0.0, 0.0).into_py(py));
        if let Some(center) = center_of_mass {
            if let Ok(v) = vector3(py, center, "center_of_mass") {
                inner.center_of_mass = v;
            }
        }
        let panel_values: Vec<PyObject> = panels
            .map(|v| v.into_iter().map(|p| p.into_py(py)).collect())
            .unwrap_or_default();
        if !panel_values.is_empty() {
            Python::with_gil(|py| {
                inner.panels = panel_values
                    .iter()
                    .filter_map(|panel| panel.extract::<Py<SatellitePanel>>(py).ok())
                    .map(|panel| panel.borrow(py).native().clone())
                    .collect();
            });
        }
        Ok(Self {
            inner,
            inertia_value: inertia.into_py(py),
            center_value,
            panel_values,
        })
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
    fn inertia(&self, py: Python<'_>) -> PyObject {
        self.inertia_value.clone_ref(py)
    }
    #[setter]
    fn set_inertia(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        if let Ok(v) = matrix3(py, value) {
            self.inner.inertia = v;
        }
        self.inertia_value = value.into_py(py);
        Ok(())
    }
    #[getter]
    fn center_of_mass(&self, py: Python<'_>) -> PyObject {
        self.center_value.clone_ref(py)
    }
    #[setter]
    fn set_center_of_mass(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        if let Ok(v) = vector3(py, value, "center_of_mass") {
            self.inner.center_of_mass = v;
        }
        self.center_value = value.into_py(py);
        Ok(())
    }
    #[getter]
    fn panels(&self, py: Python<'_>) -> Vec<PyObject> {
        self.panel_values.iter().map(|p| p.clone_ref(py)).collect()
    }
    #[setter]
    fn set_panels(&mut self, _py: Python<'_>, panels: &PyAny) -> PyResult<()> {
        let values: Vec<PyObject> = panels.extract()?;
        Python::with_gil(|py| {
            self.inner.panels = values
                .iter()
                .filter_map(|panel| panel.extract::<Py<SatellitePanel>>(py).ok())
                .map(|panel| panel.borrow(py).native().clone())
                .collect();
        });
        self.panel_values = values;
        Ok(())
    }
}

impl SatelliteGeometry {
    pub fn native(&self) -> &NativeSatelliteGeometry {
        &self.inner
    }
}

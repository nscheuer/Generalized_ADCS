use super::*;

#[pyclass(name = "Satellite")]
pub(super) struct PySatellite {
    inner: NativeSatellite<PyObject, PyObject, PyObject>,
}

#[pymethods]
impl PySatellite {
    #[new]
    #[pyo3(signature = (geometry, adcs=None, state_machine=None))]
    fn new(
        py: Python<'_>,
        geometry: PyObject,
        adcs: Option<PyObject>,
        state_machine: Option<PyObject>,
    ) -> PyResult<Self> {
        let adcs = match adcs {
            Some(value) if !value.is_none(py) => value,
            _ => py
                .import("GenADCS.backends.rust.hardware.ADCS_subsystem")?
                .getattr("ADCSSubsystem")?
                .call0()?
                .into_py(py),
        };
        let state_machine = state_machine.unwrap_or_else(|| py.None());
        Ok(Self {
            inner: NativeSatellite::new(geometry, adcs, state_machine),
        })
    }

    #[getter]
    fn geometry(&self, py: Python<'_>) -> PyObject {
        self.inner.geometry.clone_ref(py)
    }

    #[setter]
    fn set_geometry(&mut self, geometry: PyObject) {
        self.inner.geometry = geometry;
    }

    #[getter]
    fn adcs(&self, py: Python<'_>) -> PyObject {
        self.inner.adcs.clone_ref(py)
    }

    #[setter]
    fn set_adcs(&mut self, adcs: PyObject) {
        self.inner.adcs = adcs;
    }

    #[getter]
    fn state_machine(&self, py: Python<'_>) -> PyObject {
        self.inner.state_machine.clone_ref(py)
    }

    #[setter]
    fn set_state_machine(&mut self, state_machine: PyObject) {
        self.inner.state_machine = state_machine;
    }
}

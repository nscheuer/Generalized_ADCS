use pyo3::prelude::*;

#[pymodule]
fn _state(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<crate::attitude_state_py::AttitudeState>()?;
    module.add_class::<crate::internal_state_py::InternalState>()?;
    module.add_class::<crate::internal_state_py::WheelMomentumState>()?;
    module.add_class::<crate::orbital_state_py::OrbitalState>()?;
    module.add_class::<crate::state_py::State>()?;
    Ok(())
}

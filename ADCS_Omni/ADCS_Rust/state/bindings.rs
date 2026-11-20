use pyo3::prelude::*;

#[pymodule]
fn _state(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<crate::attitude_state::AttitudeState>()?;
    module.add_class::<crate::internal_state::InternalState>()?;
    module.add_class::<crate::internal_state::WheelMomentumState>()?;
    module.add_class::<crate::orbital_state::OrbitalState>()?;
    module.add_class::<crate::state::State>()?;
    Ok(())
}

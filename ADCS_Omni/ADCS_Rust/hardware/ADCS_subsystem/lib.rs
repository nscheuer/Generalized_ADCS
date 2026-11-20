use pyo3::prelude::*;

include!("effects.rs");
include!("magnetorquer.rs");
include!("reaction_wheel.rs");
include!("adcs_subsystem.rs");

#[pymodule]
fn _subsystem(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_effects, m)?)?;
    m.add_class::<Magnetorquer>()?;
    m.add_class::<ReactionWheel>()?;
    m.add_class::<ADCSSubsystem>()?;
    Ok(())
}

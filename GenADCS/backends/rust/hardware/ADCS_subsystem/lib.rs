#![allow(non_local_definitions)]
use pyo3::prelude::*;

include!("effects.rs");
include!("magnetorquer.rs");
include!("reaction_wheel.rs");
include!("adcs_subsystem.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_adcs_subsystem.rs"]
mod test_adcs_subsystem;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_effects.rs"]
mod test_effects;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_magnetorquer.rs"]
mod test_magnetorquer;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/ADCS_subsystem/test_reaction_wheel.rs"]
mod test_reaction_wheel;

#[pymodule]
fn _subsystem(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_effects, m)?)?;
    m.add_class::<Magnetorquer>()?;
    m.add_class::<ReactionWheel>()?;
    m.add_class::<ADCSSubsystem>()?;
    Ok(())
}

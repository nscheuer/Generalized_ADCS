#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod dynamics_config;
include!("dynamics_config_py.rs");
mod simulation_config;
include!("simulation_config_py.rs");

#[cfg(test)]
#[path = "../../../tests/backends/rust/config/test_dynamics_config.rs"]
mod test_dynamics_config;

#[cfg(test)]
#[path = "../../../tests/backends/rust/config/test_simulation_config.rs"]
mod test_simulation_config;

#[pymodule]
fn _config(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DynamicsConfig>()?;
    m.add_class::<SimulationConfig>()?;
    Ok(())
}

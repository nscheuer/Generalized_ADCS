use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

include!("dynamics_config.rs");
include!("simulation_config.rs");

#[pymodule]
fn _config(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DynamicsConfig>()?;
    m.add_class::<SimulationConfig>()?;
    Ok(())
}

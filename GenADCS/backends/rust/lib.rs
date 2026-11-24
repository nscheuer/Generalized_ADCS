#![allow(non_local_definitions)]
use pyo3::prelude::*;
pub mod simulate;
mod simulate_py;
pub use simulate::SimulationTrajectory;
#[pymodule]
fn _simulate(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let format = py
        .import("GenADCS.backends.shared.simulationresult.resultsformat")?
        .getattr("ResultsFormat")?;
    m.add("ResultsFormat", format)?;
    m.add_function(wrap_pyfunction!(simulate_py::simulate, m)?)?;
    Ok(())
}

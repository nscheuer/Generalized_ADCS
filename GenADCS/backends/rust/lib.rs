#![allow(non_local_definitions)]
use pyo3::exceptions::{PyTypeError,PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict,PySlice,PyTuple};
include!("simulate.rs");
#[pymodule]
fn _simulate(py:Python<'_>,m:&PyModule)->PyResult<()>{
    let format = py.import("GenADCS.backends.shared.simulationresult.resultsformat")?.getattr("ResultsFormat")?;
    m.add("ResultsFormat", format)?;
    m.add_function(wrap_pyfunction!(simulate,m)?)?;
    Ok(())
}

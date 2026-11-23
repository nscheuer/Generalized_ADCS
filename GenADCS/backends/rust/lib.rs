#![allow(non_local_definitions)]
use pyo3::exceptions::{PyTypeError,PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict,PySlice,PyTuple};
include!("helpers/input_output/resultsformat.rs");
include!("simulate.rs");
#[pymodule]
fn _simulate(_py:Python<'_>,m:&PyModule)->PyResult<()>{m.add_class::<ResultsFormat>()?;m.add_function(wrap_pyfunction!(simulate,m)?)?;Ok(())}

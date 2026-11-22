#![allow(non_local_definitions)]
use pyo3::prelude::*;
include!("satellite.rs");
#[pymodule]
fn _satellite(_py:Python<'_>,m:&PyModule)->PyResult<()>{m.add_class::<Satellite>()?;Ok(())}

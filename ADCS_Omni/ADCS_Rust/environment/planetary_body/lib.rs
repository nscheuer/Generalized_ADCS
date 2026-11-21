use pyo3::prelude::*;

include!("planetary_body.rs");
include!("earth.rs");

#[pymodule]
fn _planetary_body(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PlanetaryBody>()?;
    m.add_class::<Earth>()?;
    Ok(())
}

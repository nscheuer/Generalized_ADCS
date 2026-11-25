#![allow(non_local_definitions)]
use pyo3::prelude::*;

include!("_coupled_dynamics.rs");
include!("_dop853_coefficients.rs");
include!("coupled_rk4_propagator.rs");
include!("coupled_dop853_propagator.rs");
include!("integrate.rs");

#[path = "_coupled_dynamics_py.rs"]
mod dynamics_py;
#[path = "_coupled_propagators_py.rs"]
mod propagators_py;

#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/sensors/kernels.rs"]
mod sensor_kernels;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/base.rs"]
mod base;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_mekf.rs"]
mod mekf;
mod sensor_sampling;

#[pymodule]
fn _coupled(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dynamics_py::derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dynamics_py::project, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::rk4_step, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::dop853_attempt, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::integrate_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::integrate_dop853, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_sampling::sample_sensors, m)?)?;
    Ok(())
}

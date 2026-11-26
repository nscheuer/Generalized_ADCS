#![allow(non_local_definitions)]
use pyo3::prelude::*;

include!("_coupled_dynamics.rs");
include!("_dop853_coefficients.rs");
include!("coupled_rk4_propagator.rs");
include!("coupled_dop853_propagator.rs");
include!("integrate.rs");

#[path = "../attitude_propagators/mod.rs"]
mod attitude_propagators;
#[path = "../orbital_propagators/mod.rs"]
mod orbital_propagators;
#[path = "../separate_solver.rs"]
mod separate_solver;

#[path = "_coupled_dynamics_py.rs"]
mod dynamics_py;
#[path = "_coupled_propagators_py.rs"]
mod propagators_py;
#[path = "../separate_solver_py.rs"]
mod separate_solver_py;

#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/base.rs"]
mod base;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/direction_correction.rs"]
mod direction_correction;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_mekf.rs"]
mod attitude_mekf;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_ekf.rs"]
mod attitude_ekf;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_sequential_ukf.rs"]
mod attitude_sequential_ukf;
#[allow(dead_code)]
#[path = "../../hardware/ADCS_subsystem/sensors/kernels.rs"]
mod sensor_kernels;
mod sensor_sampling;
mod simulation_sampling;

#[pymodule]
fn _coupled(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dynamics_py::derivative, m)?)?;
    m.add_function(wrap_pyfunction!(dynamics_py::project, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::rk4_step, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::dop853_attempt, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::integrate_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::integrate_dop853, m)?)?;
    m.add_function(wrap_pyfunction!(propagators_py::simulate_coupled, m)?)?;
    m.add_function(wrap_pyfunction!(
        attitude_propagators::rk4_attitude_propagator_py::rk4_attitude_step,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        attitude_propagators::reference_attitude_propagator_py::set_reference_attitude,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(separate_solver_py::integrate_uncoupled, m)?)?;
    m.add_function(wrap_pyfunction!(separate_solver_py::integrate_separate, m)?)?;
    m.add_function(wrap_pyfunction!(separate_solver_py::simulate_uncoupled, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_sampling::sample_sensors, m)?)?;
    orbital_propagators::keplerian_orbital_propagator_py::register_keplerian(m)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_keplerian_orbital_propagator.rs"]
mod test_keplerian_orbital_propagator;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_rk4_attitude_propagator.rs"]
mod test_rk4_attitude_propagator;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_reference_attitude_propagator.rs"]
mod test_reference_attitude_propagator;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_coupled_rk4_propagator.rs"]
mod test_coupled_rk4_propagator;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_coupled_dop853_propagator.rs"]
mod test_coupled_dop853_propagator;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/propagation/test_uncoupled.rs"]
mod test_uncoupled;

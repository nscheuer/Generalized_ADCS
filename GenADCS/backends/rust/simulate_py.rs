use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
#[pyo3(signature=(satellite,environment,dynamics,state,solver,backend="Rust",resultsformat=None,seed=0))]
pub fn simulate(
    py: Python<'_>,
    satellite: PyObject,
    environment: PyObject,
    dynamics: PyObject,
    mut state: PyObject,
    mut solver: PyObject,
    backend: &str,
    resultsformat: Option<PyObject>,
    seed: u64,
) -> PyResult<PyObject> {
    if backend != "Rust" {
        return Err(PyValueError::new_err("backend must be Rust"));
    }
    let solver_module = py.import("GenADCS.backends.rust.solver._solver")?;
    if solver
        .as_ref(py)
        .is_instance(solver_module.getattr("Solver")?)?
    {
        solver = solver.as_ref(py).getattr("coupled")?.into_py(py);
    }
    if solver.is_none(py)
        || !solver
            .as_ref(py)
            .is_instance(solver_module.getattr("CoupledSolver")?)?
    {
        return Err(PyTypeError::new_err(
            "solver must be a CoupledSolver or a Solver containing one",
        ));
    }
    let rust_states = py.import("GenADCS.backends.rust.state")?;
    if !state
        .as_ref(py)
        .is_instance(rust_states.getattr("State")?)?
    {
        state = rust_states
            .getattr("State")?
            .call_method1("from_python", (state,))?
            .into_py(py);
    }
    let solver_ref = solver.as_ref(py);
    let tf: f64 = solver_ref.getattr("tf")?.extract()?;
    let mut format = match resultsformat {
        Some(value) => value,
        None => py
            .import("GenADCS.backends.shared.simulationresult.resultsformat")?
            .getattr("ResultsFormat")?
            .call0()?
            .into_py(py),
    };
    let adcs = satellite.as_ref(py).getattr("adcs")?;
    let sensor_n: usize = adcs.getattr("N")?.extract()?;
    if sensor_n == 0 {
        return Err(PyValueError::new_err("ADCSSubsystem.N must be positive"));
    }
    let estimator = adcs.getattr("estimator")?.into_py(py);
    if !estimator.is_none(py) {
        let flags = PyDict::new(py);
        for name in estimator.as_ref(py).getattr("record")?.iter()? {
            flags.set_item(name?, true)?;
        }
        format = py
            .import("dataclasses")?
            .getattr("replace")?
            .call((format,), Some(flags))?
            .into_py(py);
    }
    let n: usize = format.as_ref(py).getattr("N")?.extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("progress", solver_ref.getattr("progress")?)?;
    kwargs.set_item(
        "progress_chunk_steps",
        solver_ref.getattr("progress_chunk_steps")?,
    )?;
    if n == 0 {
        return Err(PyValueError::new_err("ResultsFormat.N must be positive"));
    }
    let stride = if estimator.is_none(py) {
        n
    } else {
        let (mut a, mut b) = (n, sensor_n);
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    };
    kwargs.set_item("record_every", stride)?;
    let environment_for_record = environment.clone_ref(py);
    let dynamics_for_record = dynamics.clone_ref(py);
    let satellite_for_record = satellite.clone_ref(py);
    let result = solver_ref.getattr("propagator")?.call_method(
        "integrate",
        (state.clone_ref(py), satellite, environment, dynamics, tf),
        Some(kwargs),
    )?;
    let trajectory = result.downcast::<PyTuple>().map_err(|_| {
        PyValueError::new_err("propagator.integrate must return times and state vectors")
    })?;
    if trajectory.len() != 2 {
        return Err(PyValueError::new_err(
            "propagator.integrate must return times and state vectors",
        ));
    }
    let times = trajectory.get_item(0)?.into_py(py);
    let numpy = py.import("numpy")?;
    let vectors = numpy
        .call_method1("asarray", (trajectory.get_item(1)?,))?
        .call_method1("astype", ("float64",))?;
    let shape: Vec<usize> = vectors.getattr("shape")?.extract()?;
    let time_values: Vec<f64> = numpy
        .call_method1("asarray", (times.as_ref(py),))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    let state_values: Vec<Vec<f64>> = vectors.call_method0("tolist")?.extract()?;
    let trajectory = crate::simulate::SimulationTrajectory::new(time_values, state_values)
        .map_err(PyValueError::new_err)?;
    if shape.len() != 2 || shape[0] != trajectory.times.len() || shape[1] != trajectory.state_width
    {
        return Err(PyValueError::new_err(
            "trajectory arrays must be rectangular and aligned",
        ));
    }
    let initial_orbit = state.as_ref(py).getattr("orbit")?;
    let epoch_jd: f64 = initial_orbit.getattr("julian_date")?.extract()?;
    let interval: f64 = solver_ref
        .getattr("propagator")?
        .getattr("dt")?
        .extract::<f64>()?
        * n as f64;
    let rows: Vec<usize> = trajectory
        .times
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if (t / interval - (t / interval).round()).abs() < 1e-9 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let sampled_times = times.as_ref(py).get_item(rows.clone())?;
    let sampled_vectors = vectors.get_item(rows.clone())?;
    let prepared = py
        .import("GenADCS.backends.rust.propagation.coupled_propagators._coupled_dynamics")?
        .getattr("_prepare")?
        .call1((
            satellite_for_record.as_ref(py),
            environment_for_record.as_ref(py),
            dynamics_for_record.as_ref(py),
        ))?;
    let adcs = satellite_for_record.as_ref(py).getattr("adcs")?;
    let sensor_n: usize = adcs.getattr("N")?.extract()?;
    if sensor_n == 0 {
        return Err(PyValueError::new_err("ADCSSubsystem.N must be positive"));
    }
    let dt: f64 = solver_ref.getattr("propagator")?.getattr("dt")?.extract()?;
    let samples = py
        .import("GenADCS.backends.rust.propagation.coupled_propagators._coupled")?
        .getattr("sample_sensors")?
        .call1((
            times.as_ref(py),
            vectors,
            epoch_jd,
            dt * sensor_n as f64,
            prepared,
            adcs.getattr("sensors")?,
            environment_for_record.as_ref(py),
            seed,
            estimator,
            format
                .as_ref(py)
                .getattr("normalized_estimation_error_squared")?,
        ))?;
    let selected = PyDict::new(py);
    for (key, value) in samples.downcast::<PyDict>()? {
        selected.set_item(key, value.get_item(rows.clone())?)?;
    }
    let result = py
        .import("GenADCS.backends.shared.simulationresult.record")?
        .getattr("record_samples")?
        .call1((
            sampled_times,
            sampled_vectors,
            format,
            satellite_for_record,
            environment_for_record,
            dynamics_for_record,
            epoch_jd,
            backend,
            selected,
        ))?;
    Ok(result.into_py(py))
}

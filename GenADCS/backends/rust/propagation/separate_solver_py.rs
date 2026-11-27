use numpy::{ndarray::Array2, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::control_loop_py::{attach_control_logs_to_pydict, build_control_loop_context};
use super::dynamics_py::numbers;
use super::sensor_sampling::{build_sampler_and_runtime, sampler_to_pydict};
use super::separate_solver::{integrate_uncoupled_native, integrate_uncoupled_observed};
use super::Runtime;

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,reference_mode=-1,progress=false,progress_chunk_steps=500,record_every=1))]
pub(super) fn integrate_uncoupled(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    reference_mode: i32,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
) -> PyResult<PyObject> {
    if progress_chunk_steps == 0 {
        return Err(PyValueError::new_err(
            "progress_chunk_steps must be positive",
        ));
    }
    let y = numbers(py, y)?;
    let ephemeris_path: Option<String> = if reference_mode == 2 {
        Some(runtime.getattr("ephemeris_path")?.extract()?)
    } else {
        None
    };
    let mut runtime = Runtime::from_py(py, runtime)?;
    if let Some(path) = ephemeris_path {
        if runtime.ephemeris.is_none() {
            runtime.ephemeris = Some(
                ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                    .map_err(PyValueError::new_err)?,
            );
        }
    }
    let (result, signal_error) = py.allow_threads(move || {
        let mut signal_error = None;
        let result = integrate_uncoupled_native(
            y,
            jd,
            dt,
            tf,
            &mut runtime,
            reference_mode,
            progress,
            progress_chunk_steps,
            record_every,
            || {
                Python::with_gil(|py| match py.check_signals() {
                    Ok(()) => true,
                    Err(error) => {
                        signal_error = Some(error);
                        false
                    }
                })
            },
        );
        (result, signal_error)
    });
    if let Some(error) = signal_error {
        return Err(error);
    }
    let (times, states) = result.map_err(PyValueError::new_err)?;
    let width = states.first().map_or(0, Vec::len);
    let array = Array2::from_shape_vec(
        (states.len(), width),
        states.into_iter().flatten().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((times.into_pyarray(py), array.into_pyarray(py)).into_py(py))
}

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,record_every,sensor_every,runtime,sensors,environment,seed=0,estimator=None,nees=false,reference_mode=-1,progress=false,progress_chunk_steps=500,adcs=None,obc=None,ctrl_log_flags=0,mu=3.986004418e14))]
pub(super) fn simulate_uncoupled(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    record_every: usize,
    sensor_every: usize,
    runtime: &PyAny,
    sensors: &PyAny,
    environment: &PyAny,
    seed: u64,
    estimator: Option<&PyAny>,
    nees: bool,
    reference_mode: i32,
    progress: bool,
    progress_chunk_steps: usize,
    adcs: Option<&PyAny>,
    obc: Option<&PyAny>,
    ctrl_log_flags: u32,
    mu: f64,
) -> PyResult<PyObject> {
    if progress_chunk_steps == 0 {
        return Err(PyValueError::new_err(
            "progress_chunk_steps must be positive",
        ));
    }
    let y = numbers(py, y)?;
    let (mut sampler, mut runtime) = build_sampler_and_runtime(
        py,
        jd,
        runtime,
        sensors,
        environment,
        seed,
        estimator,
        nees,
        reference_mode == 2,
    )?;
    let sensor_interval = dt * sensor_every as f64;
    let mut ctrl_ctx = build_control_loop_context(
        py,
        &sampler,
        &runtime,
        adcs,
        obc,
        jd,
        sensor_interval,
        ctrl_log_flags,
        mu,
    )?;
    let (result, signal_error) = py.allow_threads(move || {
        let mut signal_error = None;
        let mut observe =
            |t: f64, state: &[f64], recorded: bool, rt: &mut Runtime| -> Result<(), String> {
                let on_sensor = sensor_every > 0
                    && (t / sensor_interval - (t / sensor_interval).round()).abs() < 1e-9;
                if on_sensor {
                    let tick = (t / sensor_interval).round() as u64;
                    sampler.sample(state, t, tick, rt)?;
                    if let Some(ctx) = ctrl_ctx.as_mut() {
                        ctx.step(t, state, &mut sampler, rt);
                    }
                }
                if recorded {
                    sampler.record(on_sensor);
                    if let Some(ctx) = ctrl_ctx.as_mut() {
                        ctx.record(on_sensor);
                    }
                }
                Ok(())
            };
        let result = integrate_uncoupled_observed(
            y,
            jd,
            dt,
            tf,
            &mut runtime,
            reference_mode,
            progress,
            progress_chunk_steps,
            record_every,
            || {
                Python::with_gil(|py| match py.check_signals() {
                    Ok(()) => true,
                    Err(error) => {
                        signal_error = Some(error);
                        false
                    }
                })
            },
            &mut observe,
        );
        (
            result.map(|(times, states)| (times, states, sampler, ctrl_ctx)),
            signal_error,
        )
    });
    if let Some(error) = signal_error {
        return Err(error);
    }
    let (times, states, sampler, ctrl_ctx) = result.map_err(PyValueError::new_err)?;
    let width = states.first().map_or(0, Vec::len);
    let array = Array2::from_shape_vec(
        (states.len(), width),
        states.into_iter().flatten().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let samples_dict = sampler_to_pydict(py, &sampler)?;
    if let Some(ctx) = ctrl_ctx.as_ref() {
        attach_control_logs_to_pydict(py, samples_dict.as_ref(py), ctx)?;
    }
    Ok((times.into_pyarray(py), array.into_pyarray(py), samples_dict).into_py(py))
}

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,reference_mode=-1,progress=false,progress_chunk_steps=500,record_every=1))]
pub(super) fn integrate_separate(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    reference_mode: i32,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
) -> PyResult<PyObject> {
    integrate_uncoupled(
        py,
        y,
        jd,
        dt,
        tf,
        runtime,
        reference_mode,
        progress,
        progress_chunk_steps,
        record_every,
    )
}

fn output(py: Python<'_>, times: Vec<f64>, states: Vec<Vec<f64>>) -> PyResult<PyObject> {
    let np = py.import("numpy")?;
    let ys = np.call_method1("asarray", (states,))?;
    Ok((times.to_pyarray(py), ys).into_py(py))
}

struct NativeProgress {
    label: &'static str,
    total: Option<usize>,
    simulation_total: Option<f64>,
    simulation_elapsed: f64,
    main_dt: f64,
    steps: usize,
    chunk: usize,
    pending: usize,
    started: std::time::Instant,
    last_draw: std::time::Instant,
    width: usize,
    finished: bool,
}
impl NativeProgress {
    fn new(
        label: &'static str,
        total: Option<usize>,
        simulation_total: Option<f64>,
        main_dt: f64,
        chunk: usize,
    ) -> Self {
        let now = std::time::Instant::now();
        let bar = Self {
            label,
            total,
            simulation_total,
            simulation_elapsed: 0.,
            main_dt,
            steps: 0,
            chunk: chunk.max(1),
            pending: 0,
            started: now,
            last_draw: now,
            width: 30,
            finished: false,
        };
        bar.draw(false);
        bar
    }
    fn advance(&mut self, simulated_seconds: f64) {
        self.simulation_elapsed += simulated_seconds;
        self.steps = ((self.simulation_elapsed / self.main_dt) + 1e-12).floor() as usize;
        self.pending += 1;
        if self.pending >= self.chunk || self.last_draw.elapsed().as_millis() >= 100 {
            self.pending = 0;
            self.last_draw = std::time::Instant::now();
            self.draw(false);
        }
    }
    fn finish(&mut self) {
        if !self.finished {
            self.draw(true);
            self.finished = true;
        }
    }
    fn draw(&self, done: bool) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let rate = if elapsed > 0. {
            self.steps as f64 / elapsed
        } else {
            0.
        };
        if let Some(total) = self.total {
            let fraction = if total > 0 {
                self.steps as f64 / total as f64
            } else {
                1.
            };
            let filled = (fraction.clamp(0., 1.) * self.width as f64).round() as usize;
            eprint!(
                "\r{}: [{}{}] {:>3.0}% ({}/{}) [{:.0} step/s]",
                self.label,
                "=".repeat(filled),
                " ".repeat(self.width - filled),
                100. * fraction,
                self.steps,
                total,
                rate
            );
        } else if let Some(total) = self.simulation_total {
            let fraction = (self.simulation_elapsed / total).clamp(0., 1.);
            let filled = (fraction * self.width as f64).round() as usize;
            eprint!(
                "\r{}: [{}{}] {:>3.0}% ({:.1}/{:.1} sim s, {} steps) [{:.0} step/s]",
                self.label,
                "=".repeat(filled),
                " ".repeat(self.width - filled),
                100. * fraction,
                self.simulation_elapsed,
                total,
                self.steps,
                rate
            );
        }
        if done {
            eprintln!();
        }
    }
}
/// Integrate an already constructed Rust runtime without entering Python.
pub fn integrate_rk4_native(
    mut y: Vec<f64>,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &Runtime,
    progress: bool,
    progress_chunk_steps: usize,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if !tf.is_finite() || tf < 0. {
        return Err(PyValueError::new_err(
            "tf must be finite and nonnegative seconds",
        ));
    }
    if !dt.is_finite() || dt <= 0. {
        return Err(PyValueError::new_err(
            "dt must be finite and positive seconds",
        ));
    }
    if y.len() != 13 {
        return Err(PyValueError::new_err(
            "coupled state must contain 13 values",
        ));
    }
    normalize_q(&mut y)?;
    let mut times = vec![0.];
    let mut states = vec![y.clone()];
    let full = (tf / dt).floor() as usize;
    let mut remainder = tf - full as f64 * dt;
    if remainder <= 1e-12_f64 * tf.max(1.) {
        remainder = 0.;
    }
    let mut bar = if progress {
        Some(NativeProgress::new(
            "Running coupled RK4",
            Some(full + usize::from(remainder > 0.)),
            None,
            dt,
            progress_chunk_steps,
        ))
    } else {
        None
    };
    for i in 0..(full + usize::from(remainder > 0.)) {
        let elapsed = i as f64 * dt;
        let h = if i == full { remainder } else { dt };
        y = rk4_vec(&y, jd + elapsed / 86400., h, runtime)?;
        times.push(((i + 1) as f64 * dt).min(tf));
        states.push(y.clone());
        if let Some(bar) = bar.as_mut() {
            bar.advance(h);
        }
    }
    *times.last_mut().unwrap() = tf;
    if let Some(bar) = bar.as_mut() {
        bar.finish();
    }
    Ok((times, states))
}

/// Adaptive DOP853 propagation using only Rust model and solver code.
pub fn integrate_dop853_native(
    mut y: Vec<f64>,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &Runtime,
    rtol: f64,
    mut atol: Vec<f64>,
    max_step: f64,
    min_step: f64,
    progress: bool,
    progress_chunk_steps: usize,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if !tf.is_finite() || tf < 0. {
        return Err(PyValueError::new_err(
            "tf must be finite and nonnegative seconds",
        ));
    }
    if !dt.is_finite() || dt <= 0. {
        return Err(PyValueError::new_err(
            "dt must be finite and positive seconds",
        ));
    }
    if y.len() != 13 {
        return Err(PyValueError::new_err(
            "coupled state must contain 13 values",
        ));
    }
    normalize_q(&mut y)?;
    if atol.len() == 1 {
        atol = vec![atol[0]; y.len()];
    }
    if atol.len() != y.len() {
        return Err(PyValueError::new_err("atol length differs from state size"));
    }
    let mut times = vec![0.];
    let mut states = vec![y.clone()];
    if tf == 0. {
        return Ok((times, states));
    }
    let mut t = 0.;
    let mut h = dt.min(max_step);
    let mut f = derivative_vec(&y, jd, runtime)?;
    let mut trials = 0usize;
    let mut bar = if progress {
        Some(NativeProgress::new(
            "Running coupled DOP853",
            Some((tf / dt).ceil() as usize),
            None,
            dt,
            progress_chunk_steps,
        ))
    } else {
        None
    };
    while t < tf {
        trials += 1;
        if trials > 10_000_000 {
            return Err(PyValueError::new_err("DOP853 exceeded maximum trials"));
        }
        let remaining = tf - t;
        let step = h.min(max_step).min(remaining);
        if step <= 0. || t + step == t {
            return Err(PyValueError::new_err(
                "DOP853 step is too small to advance simulation time",
            ));
        }
        if step < min_step && remaining > min_step {
            return Err(PyValueError::new_err(
                "DOP853 reached min_step before satisfying tolerances",
            ));
        }
        let (candidate, next_f, error) =
            dop853_vec(&y, &f, jd + t / 86400., step, runtime, rtol, &atol)?;
        if !error.is_finite() {
            return Err(PyValueError::new_err(
                "DOP853 produced a nonfinite error estimate",
            ));
        }
        if error <= 1. {
            t += step;
            y = candidate;
            f = next_f;
            times.push(t);
            states.push(y.clone());
            if let Some(bar) = bar.as_mut() {
                bar.advance(step);
            }
            let factor = if error == 0. {
                10.
            } else {
                (0.9 * error.powf(-0.125)).clamp(0.2, 10.)
            };
            h = max_step.min(step * factor);
        } else {
            h = step * (0.9 * error.powf(-0.125)).max(0.2);
            if h < min_step {
                return Err(PyValueError::new_err(
                    "DOP853 reached min_step before satisfying tolerances",
                ));
            }
        }
    }
    if let Some(bar) = bar.as_mut() {
        bar.finish();
    }
    Ok((times, states))
}
#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,progress=false,progress_chunk_steps=500))]
fn integrate_rk4(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    progress: bool,
    progress_chunk_steps: usize,
) -> PyResult<PyObject> {
    if !tf.is_finite() || tf < 0. {
        return Err(PyValueError::new_err(
            "tf must be finite and nonnegative seconds",
        ));
    }
    if !dt.is_finite() || dt <= 0. {
        return Err(PyValueError::new_err(
            "dt must be finite and positive seconds",
        ));
    }
    let y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let (times, states) = py.allow_threads(move || {
        integrate_rk4_native(y, jd, dt, tf, &runtime, progress, progress_chunk_steps)
    })?;
    output(py, times, states)
}
#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,rtol,atol,max_step,min_step,progress=false,progress_chunk_steps=500))]
fn integrate_dop853(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    rtol: f64,
    atol: &PyAny,
    max_step: f64,
    min_step: f64,
    progress: bool,
    progress_chunk_steps: usize,
) -> PyResult<PyObject> {
    if !tf.is_finite() || tf < 0. {
        return Err(PyValueError::new_err(
            "tf must be finite and nonnegative seconds",
        ));
    }
    if !dt.is_finite() || dt <= 0. {
        return Err(PyValueError::new_err(
            "dt must be finite and positive seconds",
        ));
    }
    let y = numbers(py, y)?;
    let atol = numbers(py, atol)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let (times, states) = py.allow_threads(move || {
        integrate_dop853_native(
            y,
            jd,
            dt,
            tf,
            &runtime,
            rtol,
            atol,
            max_step,
            min_step,
            progress,
            progress_chunk_steps,
        )
    })?;
    output(py, times, states)
}

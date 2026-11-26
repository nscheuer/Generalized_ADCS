use super::{normalize_q, NativeProgress, Runtime};

pub(super) fn integrate_separate_native(
    mut y: Vec<f64>,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &Runtime,
    reference_mode: i32,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
    mut poll: impl FnMut() -> bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive seconds".into());
    }
    if !tf.is_finite() || tf < 0.0 {
        return Err("tf must be finite and nonnegative seconds".into());
    }
    if y.len() != 13 || y.iter().any(|v| !v.is_finite()) {
        return Err("separate state must contain 13 finite values".into());
    }
    if record_every == 0 {
        return Err("record_every must be a positive integer".into());
    }
    if reference_mode < -1 || reference_mode > 2 {
        return Err("reference_mode must be -1, 0, 1, or 2".into());
    }
    normalize_q(&mut y)?;
    if reference_mode >= 0 {
        super::attitude_propagators::reference_attitude_propagator::set_reference(
            &mut y,
            jd,
            reference_mode,
            runtime,
        )?;
    }
    let full = (tf / dt).floor() as usize;
    let mut remainder = tf - full as f64 * dt;
    if remainder <= 1e-12 * tf.max(1.0) {
        remainder = 0.0;
    }
    let total = full + usize::from(remainder > 0.0);
    let mut times = vec![0.0];
    let mut states = vec![y.clone()];
    let mut bar = if progress {
        Some(NativeProgress::new(
            "Running Keplerian + attitude",
            Some(total),
            None,
            dt,
            progress_chunk_steps,
        ))
    } else {
        None
    };
    for row in 0..total {
        if row % progress_chunk_steps.max(1) == 0 && !poll() {
            return Err("simulation interrupted".into());
        }
        let elapsed = row as f64 * dt;
        let step_dt = if row == full { remainder } else { dt };
        let epoch = jd + elapsed / 86400.0;
        if reference_mode < 0 {
            y = super::attitude_propagators::rk4_attitude_propagator::attitude_rk4(
                &y, epoch, step_dt, runtime,
            )?;
        }
        let (position, velocity) =
            super::orbital_propagators::keplerian_orbital_propagator::keplerian_step_native(
                [y[7], y[8], y[9]],
                [y[10], y[11], y[12]],
                runtime.mu,
                step_dt,
            )?;
        y[7..10].copy_from_slice(&position);
        y[10..13].copy_from_slice(&velocity);
        if reference_mode >= 0 {
            super::attitude_propagators::reference_attitude_propagator::set_reference(
                &mut y,
                epoch + step_dt / 86400.0,
                reference_mode,
                runtime,
            )?;
        }
        let completed = row + 1;
        if completed % record_every == 0 || completed == total {
            times.push((completed as f64 * dt).min(tf));
            states.push(y.clone());
        }
        if let Some(bar) = bar.as_mut() {
            bar.advance(step_dt);
        }
    }
    if let Some(bar) = bar.as_mut() {
        bar.finish();
    }
    *times.last_mut().expect("initial time is present") = tf;
    Ok((times, states))
}

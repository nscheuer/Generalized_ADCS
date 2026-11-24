/// Rust-native configuration for coupled propagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoupledSolverConfig {
    pub final_time: f64,
    pub progress: bool,
    pub progress_chunk_steps: usize,
}

impl CoupledSolverConfig {
    pub fn new(
        final_time: f64,
        progress: bool,
        progress_chunk_steps: usize,
    ) -> Result<Self, String> {
        if !final_time.is_finite() || final_time < 0.0 {
            return Err("tf must be finite and nonnegative seconds".to_owned());
        }
        if progress_chunk_steps == 0 {
            return Err("progress_chunk_steps must be a positive integer".to_owned());
        }
        Ok(Self {
            final_time,
            progress,
            progress_chunk_steps,
        })
    }
}

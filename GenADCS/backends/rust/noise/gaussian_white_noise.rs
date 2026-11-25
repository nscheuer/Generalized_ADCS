//! Rust-native independent Gaussian white noise.
use super::noise::standard_normals;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianWhiteNoise {
    pub standard_deviation: f64,
}
impl GaussianWhiteNoise {
    pub fn new(standard_deviation: f64) -> Result<Self, &'static str> {
        if !standard_deviation.is_finite() || standard_deviation < 0. {
            return Err("standard_deviation must be finite and non-negative");
        }
        Ok(Self { standard_deviation })
    }
    pub fn apply_with_draw(&self, value: &[f64], draw: &[f64]) -> Result<Vec<f64>, &'static str> {
        if value.len() != draw.len() {
            return Err("draw length must match value length");
        }
        Ok(value
            .iter()
            .zip(draw)
            .map(|(v, z)| v + self.standard_deviation * z)
            .collect())
    }
    pub fn apply(&self, value: &[f64], _dt: f64) -> Vec<f64> {
        static TICK: AtomicU64 = AtomicU64::new(0);
        let z = standard_normals(0, TICK.fetch_add(1, Ordering::Relaxed), value.len());
        value
            .iter()
            .zip(z)
            .map(|(v, n)| v + self.standard_deviation * n)
            .collect()
    }
}

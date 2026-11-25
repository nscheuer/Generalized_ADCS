//! Rust-native covariance noise model.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Noise {
    pub covariance: Option<Vec<Vec<f64>>>,
    pub cholesky_factor: Option<Vec<Vec<f64>>>,
}

impl Noise {
    pub fn new(covariance: Option<Vec<Vec<f64>>>) -> Result<Self, &'static str> {
        let Some(cov) = covariance else {
            return Ok(Self {
                covariance: None,
                cholesky_factor: None,
            });
        };
        let n = cov.len();
        if n == 0
            || cov.iter().any(|r| r.len() != n)
            || cov.iter().flatten().any(|x| !x.is_finite())
        {
            return Err("covariance must be a finite non-empty square matrix");
        }
        let mut l = vec![vec![0.; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut x = cov[i][j];
                for k in 0..j {
                    x -= l[i][k] * l[j][k];
                }
                if i == j {
                    if x <= 0. {
                        return Err("covariance must be positive definite");
                    }
                    l[i][j] = x.sqrt();
                } else {
                    l[i][j] = x / l[j][j];
                }
            }
        }
        Ok(Self {
            covariance: Some(cov),
            cholesky_factor: Some(l),
        })
    }
    pub fn gaussian_sample(&self, seed: u64, tick: u64) -> Result<Vec<f64>, &'static str> {
        let l = self
            .cholesky_factor
            .as_ref()
            .ok_or("configure a covariance or use a concrete noise model")?;
        let z = standard_normals(seed, tick, l.len());
        Ok((0..l.len())
            .map(|i| (0..=i).map(|j| l[i][j] * z[j]).sum())
            .collect())
    }
    pub fn apply_with_draw(&self, value: &[f64], draw: &[f64]) -> Result<Vec<f64>, &'static str> {
        let l = self
            .cholesky_factor
            .as_ref()
            .ok_or("configure a covariance or use a concrete noise model")?;
        if value.len() != l.len() || draw.len() != l.len() {
            return Err("value length must match noise covariance");
        }
        Ok((0..l.len())
            .map(|i| value[i] + (0..=i).map(|j| l[i][j] * draw[j]).sum::<f64>())
            .collect())
    }
    pub fn apply(&self, value: &[f64], _dt: f64) -> Result<Vec<f64>, &'static str> {
        static TICK: AtomicU64 = AtomicU64::new(0);
        let tick = TICK.fetch_add(1, Ordering::Relaxed);
        let sample = self.gaussian_sample(0, tick)?;
        if value.len() != sample.len() {
            return Err("value length must match noise covariance");
        }
        Ok(value.iter().zip(sample).map(|(v, n)| v + n).collect())
    }
}

pub(crate) fn standard_normals(seed: u64, tick: u64, n: usize) -> Vec<f64> {
    let mut state = seed ^ tick.wrapping_mul(0x9e3779b97f4a7c15);
    let mut uni = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u = uni().max(f64::MIN_POSITIVE);
        let v = uni();
        let r = (-2. * u.ln()).sqrt();
        out.push(r * (std::f64::consts::TAU * v).cos());
        if out.len() < n {
            out.push(r * (std::f64::consts::TAU * v).sin());
        }
    }
    out
}

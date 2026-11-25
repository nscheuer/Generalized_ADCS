//! Rust-native six-channel inertial GNSS model.
use super::kernels;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct GNSS {
    pub noise: Vec<Vec<f64>>,
    pub noise_cholesky: Vec<Vec<f64>>,
    pub bias: [f64; 6],
    pub enabled: bool,
    tick: AtomicU64,
}

impl GNSS {
    pub fn new(noise: &[Vec<f64>], bias: &[f64]) -> Result<Self, &'static str> {
        if noise.len() != 6 || noise.iter().any(|r| r.len() != 6) || bias.len() != 6 {
            return Err("noise must be 6x6 and bias must have 6 components");
        }
        let mut l = vec![vec![0.; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let mut x = noise[i][j];
                for k in 0..j {
                    x -= l[i][k] * l[j][k]
                }
                if i == j {
                    if !x.is_finite() || x <= 0. {
                        return Err("noise covariance must be positive definite");
                    }
                    l[i][j] = x.sqrt()
                } else {
                    l[i][j] = x / l[j][j]
                }
            }
        }
        Ok(Self {
            noise: noise.to_vec(),
            noise_cholesky: l,
            bias: std::array::from_fn(|i| bias[i]),
            enabled: true,
            tick: AtomicU64::new(0),
        })
    }
    pub fn noiseless_biasfree_measurement(&self, p: [f64; 3], v: [f64; 3]) -> [f64; 6] {
        self.measure(p, v, false, false)
    }
    pub fn noiseless_biased_measurement(&self, p: [f64; 3], v: [f64; 3]) -> [f64; 6] {
        self.measure(p, v, true, false)
    }
    pub fn noisy_biasfree_measurement(&self, p: [f64; 3], v: [f64; 3]) -> [f64; 6] {
        self.noisy_measure(p, v, false)
    }
    pub fn noisy_biased_measurement(&self, p: [f64; 3], v: [f64; 3]) -> [f64; 6] {
        self.noisy_measure(p, v, true)
    }
    fn measure(&self, p: [f64; 3], v: [f64; 3], bias: bool, noise: bool) -> [f64; 6] {
        let out = kernels::gnss_measurement(
            p,
            v,
            self.bias.to_vec(),
            self.noise_cholesky.clone(),
            bias,
            noise,
            None,
        );
        std::array::from_fn(|i| out[i])
    }
    fn noisy_measure(&self, p: [f64; 3], v: [f64; 3], bias: bool) -> [f64; 6] {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed);
        let z = philox_native::philox::standard_normals(0, tick, 0, 6);
        let out = kernels::gnss_measurement(
            p,
            v,
            self.bias.to_vec(),
            self.noise_cholesky.clone(),
            bias,
            true,
            Some(z),
        );
        std::array::from_fn(|i| out[i])
    }
    pub fn jacobian_relative_quaternion(&self) -> [[f64; 3]; 6] {
        [[0.; 3]; 6]
    }
    pub fn jacobian_relative_mrp(&self) -> [[f64; 3]; 6] {
        [[0.; 3]; 6]
    }
    pub fn jacobian_relative_angle_vector(&self) -> [[f64; 3]; 6] {
        [[0.; 3]; 6]
    }
    pub fn jacobian_position(&self) -> [[f64; 3]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i < 3 && i == j { 1. } else { 0. }))
    }
    pub fn jacobian_velocity(&self) -> [[f64; 3]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i >= 3 && i - 3 == j { 1. } else { 0. }))
    }
    pub fn jacobian_bias(&self) -> [[f64; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
    }
    pub fn availability(&self) -> bool {
        self.enabled
    }
}

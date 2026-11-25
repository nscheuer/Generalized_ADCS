//! Rust-native autonomous star tracker.
use super::{kernels, Sensor};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AutonomousStarTracker {
    pub sensor: Sensor,
    pub noise_cholesky: [[f64; 3]; 3],
    pub fov_half_angle: f64,
    tick: AtomicU64,
}

impl AutonomousStarTracker {
    pub fn new(
        position: &[f64],
        orientation: &[f64],
        misalignment: &[f64],
        saturation_value: f64,
        noise: &[Vec<f64>],
        bias: &[f64],
        fov_half_angle: f64,
    ) -> Result<Self, &'static str> {
        if !fov_half_angle.is_finite()
            || fov_half_angle <= 0.
            || fov_half_angle > std::f64::consts::PI
        {
            return Err("fov_half_angle must be in (0, pi]");
        }
        let sensor = Sensor::new(
            position,
            orientation,
            misalignment,
            saturation_value,
            noise,
            bias,
        )?;
        let l = cholesky(sensor.noise)?;
        Ok(Self {
            sensor,
            noise_cholesky: l,
            fov_half_angle,
            tick: AtomicU64::new(0),
        })
    }
    pub fn noiseless_biasfree_measurement(&self, attitude: [f64; 4]) -> [f64; 4] {
        kernels::star_tracker_measurement(
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
            self.noise_cholesky,
            false,
            None,
        )
    }
    pub fn noiseless_biased_measurement(&self, attitude: [f64; 4]) -> [f64; 4] {
        self.noiseless_biasfree_measurement(attitude)
    }
    pub fn noisy_biasfree_measurement(&self, attitude: [f64; 4]) -> [f64; 4] {
        let t = self.tick.fetch_add(1, Ordering::Relaxed);
        let z = philox_native::philox::standard_normals(0, t, 0, 3);
        kernels::star_tracker_measurement(
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
            self.noise_cholesky,
            true,
            Some([z[0], z[1], z[2]]),
        )
    }
    pub fn noisy_biased_measurement(&self, attitude: [f64; 4]) -> [f64; 4] {
        self.noisy_biasfree_measurement(attitude)
    }
    pub fn residual(&self, measured: [f64; 4], estimated: [f64; 4]) -> [f64; 3] {
        kernels::star_tracker_residual(measured, estimated)
    }
    pub fn jacobian_relative_quaternion(&self) -> [[f64; 3]; 3] {
        [[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]
    }
    pub fn jacobian_relative_mrp(&self, d: [f64; 3]) -> [[f64; 3]; 3] {
        kernels::star_tracker_mrp_jacobian(d)
    }
    pub fn jacobian_relative_angle_vector(&self, d: [f64; 3]) -> [[f64; 3]; 3] {
        let a2 = d.iter().map(|x| x * x).sum::<f64>();
        let angle = a2.sqrt();
        let a = if angle < 1e-7 {
            1. - a2 / 24.
        } else {
            2. * (angle / 2.).sin() / angle
        };
        let b = if angle < 1e-7 {
            -1. / 12. + a2 / 480.
        } else {
            (angle / 2.).cos() / a2 - 2. * (angle / 2.).sin() / (angle * a2)
        };
        std::array::from_fn(|i| {
            std::array::from_fn(|j| (if i == j { a } else { 0. }) + b * d[i] * d[j])
        })
    }
    pub fn availability(
        &self,
        attitude: Option<[f64; 4]>,
        earth: Option<[f64; 3]>,
        sun: Option<[f64; 3]>,
    ) -> bool {
        if !self.sensor.enabled {
            return false;
        }
        let Some(q) = attitude else { return true };
        [earth, sun].into_iter().flatten().all(|direction| {
            let v = kernels::rotate(
                self.sensor.misalignment,
                kernels::rotate(self.sensor.orientation, kernels::rotate(q, direction)),
            );
            !(v[2] > 0. && v[0].hypot(v[1]).atan2(v[2]) < self.fov_half_angle)
        })
    }
    pub fn fisher_information_matrix(&self) -> [[f64; 3]; 3] {
        if !self.sensor.enabled {
            return [[0.; 3]; 3];
        }
        kernels::star_tracker_information(
            self.sensor.noise,
            self.sensor.orientation,
            self.sensor.misalignment,
        )
    }
}
fn cholesky(a: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], &'static str> {
    let mut l = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut x = a[i][j];
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
    Ok(l)
}

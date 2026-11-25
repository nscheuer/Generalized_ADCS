//! Native specific-force accelerometer.
use super::{kernels as k, Sensor};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Accelerometer {
    pub sensor: Sensor,
    pub noise_cholesky: k::Mat3,
}

impl Accelerometer {
    pub fn new(
        position: &[f64],
        orientation: &[f64],
        misalignment: &[f64],
        saturation_value: f64,
        noise: &[Vec<f64>],
        bias: &[f64],
    ) -> Result<Self, &'static str> {
        let sensor = Sensor::new(
            position,
            orientation,
            misalignment,
            saturation_value,
            noise,
            bias,
        )?;
        let l = k::cholesky3(sensor.noise).ok_or("noise covariance must be positive definite")?;
        Ok(Self {
            sensor,
            noise_cholesky: l,
        })
    }
    fn force(&self, a: [f64; 3], g: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        let specific = std::array::from_fn(|i| a[i] - g[i]);
        k::rotate_inertial_to_sensor(
            specific,
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
        )
    }
    pub fn noiseless_biasfree_measurement(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
    ) -> [f64; 3] {
        self.measure(a, g, attitude, false, false, None)
    }
    pub fn noiseless_biased_measurement(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
    ) -> [f64; 3] {
        self.measure(a, g, attitude, true, false, None)
    }
    pub fn noisy_biasfree_measurement(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
    ) -> [f64; 3] {
        self.measure(a, g, attitude, false, true, Some(draw3()))
    }
    pub fn noisy_biased_measurement(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
    ) -> [f64; 3] {
        self.measure(a, g, attitude, true, true, Some(draw3()))
    }
    fn measure(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
        with_bias: bool,
        with_noise: bool,
        draw: Option<[f64; 3]>,
    ) -> [f64; 3] {
        let v = std::array::from_fn(|i| a[i] - g[i]);
        k::vector_measurement(
            v,
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
            self.sensor.bias,
            self.noise_cholesky,
            self.sensor.saturation_value,
            with_bias,
            with_noise,
            draw,
        )
    }
    pub fn jacobian_relative_quaternion(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 4],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_quaternion(self.force(a, g, attitude), delta)
    }
    pub fn jacobian_relative_mrp(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 3],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_mrp(self.force(a, g, attitude), delta)
    }
    pub fn jacobian_relative_angle_vector(
        &self,
        a: [f64; 3],
        g: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 3],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_angle(self.force(a, g, attitude), delta)
    }
    pub fn jacobian_bias(&self) -> k::Mat3 {
        identity3()
    }
    pub fn availability(&self) -> bool {
        self.sensor.availability()
    }
}
fn identity3() -> k::Mat3 {
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
}
fn draw3() -> [f64; 3] {
    static TICK: AtomicU64 = AtomicU64::new(0);
    let z = philox_native::philox::standard_normals(0, TICK.fetch_add(1, Ordering::Relaxed), 0, 3);
    [z[0], z[1], z[2]]
}

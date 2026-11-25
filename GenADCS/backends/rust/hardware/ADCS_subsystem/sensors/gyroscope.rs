//! Native three-axis gyroscope.
use super::{kernels as k, Sensor};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Gyroscope {
    pub sensor: Sensor,
    pub noise_cholesky: k::Mat3,
}
impl Gyroscope {
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
    pub fn noiseless_biasfree_measurement(&self, rate: [f64; 3]) -> [f64; 3] {
        self.measure(rate, false, false, None)
    }
    pub fn noiseless_biased_measurement(&self, rate: [f64; 3]) -> [f64; 3] {
        self.measure(rate, true, false, None)
    }
    pub fn noisy_biasfree_measurement(&self, rate: [f64; 3]) -> [f64; 3] {
        self.measure(rate, false, true, Some(draw3()))
    }
    pub fn noisy_biased_measurement(&self, rate: [f64; 3]) -> [f64; 3] {
        self.measure(rate, true, true, Some(draw3()))
    }
    fn measure(
        &self,
        rate: [f64; 3],
        with_bias: bool,
        with_noise: bool,
        draw: Option<[f64; 3]>,
    ) -> [f64; 3] {
        k::rate_measurement(
            rate,
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
    pub fn jacobian_angular_velocity(&self) -> k::Mat3 {
        identity3()
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

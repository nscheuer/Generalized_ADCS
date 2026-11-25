//! Native three-axis magnetometer.
use super::{kernels as k, Sensor};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Magnetometer {
    pub sensor: Sensor,
    pub noise_cholesky: k::Mat3,
}
impl Magnetometer {
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
    pub fn noiseless_biasfree_measurement(&self, field: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        self.measure(field, attitude, false, false, None)
    }
    pub fn noiseless_biased_measurement(&self, field: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        self.measure(field, attitude, true, false, None)
    }
    pub fn noisy_biasfree_measurement(&self, field: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        self.measure(field, attitude, false, true, Some(draw3()))
    }
    pub fn noisy_biased_measurement(&self, field: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        self.measure(field, attitude, true, true, Some(draw3()))
    }
    fn measure(
        &self,
        field: [f64; 3],
        attitude: [f64; 4],
        with_bias: bool,
        with_noise: bool,
        draw: Option<[f64; 3]>,
    ) -> [f64; 3] {
        k::vector_measurement(
            field,
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
    fn field_sensor(&self, field: [f64; 3], attitude: [f64; 4]) -> [f64; 3] {
        k::rotate_inertial_to_sensor(
            field,
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
        )
    }
    pub fn jacobian_bias(&self) -> k::Mat3 {
        identity3()
    }
    pub fn jacobian_relative_quaternion(
        &self,
        field: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 4],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_quaternion(self.field_sensor(field, attitude), delta)
    }
    pub fn jacobian_relative_mrp(
        &self,
        field: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 3],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_mrp(self.field_sensor(field, attitude), delta)
    }
    pub fn jacobian_relative_angle_vector(
        &self,
        field: [f64; 3],
        attitude: [f64; 4],
        delta: [f64; 3],
    ) -> k::Mat3 {
        k::attitude_vector_jacobian_angle(self.field_sensor(field, attitude), delta)
    }
    pub fn availability(&self) -> bool {
        self.sensor.availability()
    }
    pub fn fisher_information_matrix(
        &self,
        field: [f64; 3],
        attitude: [f64; 4],
        field_covariance: k::Mat3,
    ) -> k::Mat3 {
        if !self.sensor.enabled {
            return [[0.; 3]; 3];
        }
        k::magnetometer_information(
            field,
            attitude,
            self.sensor.orientation,
            self.sensor.misalignment,
            self.sensor.noise,
            field_covariance,
        )
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

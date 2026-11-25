//! Rust-native photodiode coarse Sun sensor.

use super::kernels::{self, Mat3, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct CoarseSunSensor {
    pub position: Vec3,
    pub orientation: [f64; 4],
    pub misalignment: [f64; 4],
    pub normal: Vec3,
    pub sensitivity: f64,
    pub saturation_value: f64,
    pub noise_variance: f64,
    pub noise_std: f64,
    pub bias: f64,
    pub fov_half_angle: f64,
    pub enabled: bool,
}

impl CoarseSunSensor {
    pub fn new(
        position: Vec3,
        orientation: [f64; 4],
        misalignment: [f64; 4],
        normal: Vec3,
        sensitivity: f64,
        saturation_value: f64,
        noise_variance: f64,
        bias: f64,
        fov_half_angle: f64,
    ) -> Result<Self, &'static str> {
        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0. || sensitivity <= 0. || saturation_value <= 0. {
            return Err("normal must be nonzero; sensitivity and saturation positive");
        }
        if noise_variance <= 0.
            || fov_half_angle <= 0.
            || fov_half_angle > std::f64::consts::FRAC_PI_2
        {
            return Err("noise variance and FOV must be positive; FOV <= pi/2");
        }
        Ok(Self {
            position,
            orientation: normalize(orientation),
            misalignment: normalize(misalignment),
            normal: normal.map(|x| x / norm),
            sensitivity,
            saturation_value,
            noise_variance,
            noise_std: noise_variance.sqrt(),
            bias,
            fov_half_angle,
            enabled: true,
        })
    }
    pub fn noiseless_biasfree_measurement(&self, sun: Vec3, attitude: [f64; 4]) -> f64 {
        self.measure(sun, attitude, false, false, None)
    }
    pub fn noiseless_biased_measurement(&self, sun: Vec3, attitude: [f64; 4]) -> f64 {
        self.measure(sun, attitude, true, false, None)
    }
    pub fn noisy_biasfree_measurement(&self, sun: Vec3, attitude: [f64; 4]) -> f64 {
        self.measure(sun, attitude, false, true, Some(next_noise()))
    }
    pub fn noisy_biased_measurement(&self, sun: Vec3, attitude: [f64; 4]) -> f64 {
        self.measure(sun, attitude, true, true, Some(next_noise()))
    }
    fn measure(
        &self,
        sun: Vec3,
        attitude: [f64; 4],
        bias: bool,
        noise: bool,
        draw: Option<f64>,
    ) -> f64 {
        kernels::scalar_sun_measurement(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.sensitivity,
            self.bias,
            self.noise_std,
            self.saturation_value,
            bias,
            noise,
            draw,
        )
    }
    pub fn jacobian_bias(&self) -> f64 {
        1.0
    }
    pub fn jacobian_relative_quaternion(
        &self,
        sun: Vec3,
        attitude: [f64; 4],
        delta: [f64; 4],
    ) -> Vec3 {
        coarse_attitude_jacobian(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.sensitivity,
            Parameter::Quaternion(delta),
        )
    }
    pub fn jacobian_relative_mrp(&self, sun: Vec3, attitude: [f64; 4], delta: Vec3) -> Vec3 {
        coarse_attitude_jacobian(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.sensitivity,
            Parameter::Mrp(delta),
        )
    }
    pub fn jacobian_relative_angle_vector(
        &self,
        sun: Vec3,
        attitude: [f64; 4],
        delta: Vec3,
    ) -> Vec3 {
        coarse_attitude_jacobian(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.sensitivity,
            Parameter::Angle(delta),
        )
    }
    pub fn availability(&self, sun: Vec3, attitude: [f64; 4], visible: bool) -> bool {
        kernels::coarse_sun_available(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.fov_half_angle,
            self.enabled,
            visible,
        )
    }
    pub fn fisher_information_matrix(&self, sun: Vec3, attitude: [f64; 4], visible: bool) -> Mat3 {
        if !self.availability(sun, attitude, visible) {
            return [[0.; 3]; 3];
        }
        kernels::coarse_sun_information(
            sun,
            attitude,
            self.orientation,
            self.misalignment,
            self.normal,
            self.sensitivity,
            self.noise_variance,
        )
    }
}

enum Parameter {
    Quaternion([f64; 4]),
    Mrp(Vec3),
    Angle(Vec3),
}
fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}
fn next_noise() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TICK: AtomicU64 = AtomicU64::new(0);
    let tick = TICK.fetch_add(1, Ordering::Relaxed);
    philox_native::philox::standard_normals(0, tick, 0, 1)[0]
}
fn coarse_attitude_jacobian(
    sun: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    normal: Vec3,
    sensitivity: f64,
    param: Parameter,
) -> Vec3 {
    let body = kernels::rotate(attitude, sun);
    let mount = kernels::mounting(orientation, misalignment);
    let nb = mul3v(transpose(mount), normal);
    let (rot, derivative) = match param {
        Parameter::Quaternion(raw) => {
            let q = normalize(raw);
            let p = kernels::rotate(q, body);
            let v = [q[1], q[2], q[3]];
            let m = add3(
                scale(identity(), q[0]),
                add3(skew(v), scale(outer(v, v), 1. / q[0])),
            );
            (p, scale(matmul(skew(p), m), -2.))
        }
        Parameter::Mrp(d) => {
            let s = dot(d, d);
            let den = 1. + s;
            let q = [
                (1. - s) / den,
                2. * d[0] / den,
                2. * d[1] / den,
                2. * d[2] / den,
            ];
            let p = kernels::rotate(q, body);
            let m = add3(
                scale(identity(), 1. - s),
                add3(scale(skew(d), 2.), scale(outer(d, d), 2.)),
            );
            (p, scale(matmul(skew(p), m), -4. / den.powi(2)))
        }
        Parameter::Angle(d) => {
            let t = dot(d, d).sqrt();
            let q = if t < 1e-15 {
                [1., 0., 0., 0.]
            } else {
                let h = (t / 2.).sin() / t;
                [(t / 2.).cos(), h * d[0], h * d[1], h * d[2]]
            };
            let p = kernels::rotate(q, body);
            let j = left_jacobian(d);
            (p, scale(matmul(skew(p), j), -1.))
        }
    };
    let _ = rot;
    let mut out = [0.; 3];
    for j in 0..3 {
        out[j] = sensitivity * (0..3).map(|i| nb[i] * derivative[i][j]).sum::<f64>();
    }
    out
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn transpose(a: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn mul3v(a: Mat3, v: Vec3) -> Vec3 {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}
fn matmul(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn add3(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn scale(a: Mat3, s: f64) -> Mat3 {
    a.map(|r| r.map(|x| x * s))
}
fn identity() -> Mat3 {
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
}
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i] * b[j]))
}
fn skew(v: Vec3) -> Mat3 {
    [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]]
}
fn left_jacobian(d: Vec3) -> Mat3 {
    let t = dot(d, d).sqrt();
    let a = if t < 1e-5 {
        0.5 - t * t / 24.
    } else {
        (1. - t.cos()) / (t * t)
    };
    let b = if t < 1e-5 {
        1. / 6. - t * t / 120.
    } else {
        (t - t.sin()) / (t * t * t)
    };
    let x = skew(d);
    add3(identity(), add3(scale(x, a), scale(matmul(x, x), b)))
}

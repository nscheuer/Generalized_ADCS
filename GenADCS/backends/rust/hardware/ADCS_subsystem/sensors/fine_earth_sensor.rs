//! Rust-native two-axis fine Earth sensor.

use super::kernels::{self, Mat2, Mat3, Vec2, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct FineEarthSensor {
    pub position: Vec3,
    pub orientation: [f64; 4],
    pub misalignment: [f64; 4],
    pub noise: Mat2,
    pub noise_cholesky: Mat2,
    pub bias: Vec2,
    pub fov_half_angles: Vec2,
    pub enabled: bool,
}
impl FineEarthSensor {
    pub fn new(
        position: Vec3,
        orientation: [f64; 4],
        misalignment: [f64; 4],
        noise: Mat2,
        bias: Vec2,
        fov_half_angles: Vec2,
    ) -> Result<Self, &'static str> {
        let l = cholesky(noise)?;
        Ok(Self {
            position,
            orientation: normalize(orientation),
            misalignment: normalize(misalignment),
            noise,
            noise_cholesky: l,
            bias,
            fov_half_angles,
            enabled: true,
        })
    }
    pub fn noiseless_biasfree_measurement(&self, earth: Vec3, attitude: [f64; 4]) -> Vec2 {
        self.measure(earth, attitude, false, false, None)
    }
    pub fn noiseless_biased_measurement(&self, earth: Vec3, attitude: [f64; 4]) -> Vec2 {
        self.measure(earth, attitude, true, false, None)
    }
    pub fn noisy_biasfree_measurement(&self, earth: Vec3, attitude: [f64; 4]) -> Vec2 {
        self.measure(earth, attitude, false, true, Some(next_noise()))
    }
    pub fn noisy_biased_measurement(&self, earth: Vec3, attitude: [f64; 4]) -> Vec2 {
        self.measure(earth, attitude, true, true, Some(next_noise()))
    }
    fn measure(
        &self,
        earth: Vec3,
        attitude: [f64; 4],
        bias: bool,
        noise: bool,
        draw: Option<Vec2>,
    ) -> Vec2 {
        kernels::fine_direction_measurement(
            earth,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            bias,
            noise,
            draw,
        )
    }
    pub fn jacobian_bias(&self) -> Mat2 {
        [[1., 0.], [0., 1.]]
    }
    pub fn jacobian_relative_quaternion(
        &self,
        earth: Vec3,
        attitude: [f64; 4],
        delta: [f64; 4],
    ) -> [[f64; 3]; 2] {
        fine_attitude_jacobian(
            earth,
            attitude,
            self.orientation,
            self.misalignment,
            Parameter::Quaternion(delta),
        )
    }
    pub fn jacobian_relative_mrp(
        &self,
        earth: Vec3,
        attitude: [f64; 4],
        delta: Vec3,
    ) -> [[f64; 3]; 2] {
        fine_attitude_jacobian(
            earth,
            attitude,
            self.orientation,
            self.misalignment,
            Parameter::Mrp(delta),
        )
    }
    pub fn jacobian_relative_angle_vector(
        &self,
        earth: Vec3,
        attitude: [f64; 4],
        delta: Vec3,
    ) -> [[f64; 3]; 2] {
        fine_attitude_jacobian(
            earth,
            attitude,
            self.orientation,
            self.misalignment,
            Parameter::Angle(delta),
        )
    }
    pub fn availability(&self, earth: Vec3, attitude: [f64; 4], visible: bool) -> bool {
        kernels::fine_direction_available(
            earth,
            attitude,
            self.orientation,
            self.misalignment,
            self.fov_half_angles,
            self.enabled,
            visible,
        )
    }
    pub fn fisher_information_matrix(
        &self,
        earth: Vec3,
        attitude: [f64; 4],
        visible: bool,
    ) -> Mat3 {
        if !self.availability(earth, attitude, visible) {
            return [[0.; 3]; 3];
        }
        let angles = self.noiseless_biasfree_measurement(earth, attitude);
        fisher(angles, self.noise, self.orientation, self.misalignment)
    }
}

enum Parameter {
    Quaternion([f64; 4]),
    Mrp(Vec3),
    Angle(Vec3),
}
fn cholesky(a: Mat2) -> Result<Mat2, &'static str> {
    let mut l = [[0.; 2]; 2];
    for i in 0..2 {
        for j in 0..=i {
            let mut v = a[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k]
            }
            if i == j {
                if v <= 0. || !v.is_finite() {
                    return Err("noise covariance must be positive definite");
                }
                l[i][j] = v.sqrt()
            } else {
                l[i][j] = v / l[j][j]
            }
        }
    }
    Ok(l)
}
fn next_noise() -> Vec2 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TICK: AtomicU64 = AtomicU64::new(0);
    let tick = TICK.fetch_add(1, Ordering::Relaxed);
    let z = philox_native::philox::standard_normals(0, tick, 0, 2);
    [z[0], z[1]]
}
fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}
fn fine_attitude_jacobian(
    earth: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    parameter: Parameter,
) -> [[f64; 3]; 2] {
    let body = kernels::rotate(attitude, earth);
    let m = kernels::mounting(orientation, misalignment);
    let (perturbed, jac) = match parameter {
        Parameter::Quaternion(raw) => {
            let q = normalize(raw);
            let p = kernels::rotate(q, body);
            let v = [q[1], q[2], q[3]];
            let a = add3(
                scale(identity(), q[0]),
                add3(skew(v), scale(outer(v, v), 1. / q[0])),
            );
            (p, scale(matmul(skew(p), a), -2.))
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
            let a = add3(
                scale(identity(), 1. - s),
                add3(scale(skew(d), 2.), scale(outer(d, d), 2.)),
            );
            (p, scale(matmul(skew(p), a), -4. / den.powi(2)))
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
            let a = left_jacobian(d);
            (p, scale(matmul(skew(p), a), -1.))
        }
    };
    let sensor = mul3v(m, perturbed);
    matmul23(direction_jacobian(sensor), matmul(m, jac))
}

fn fisher(angles: Vec2, cov: Mat2, orientation: [f64; 4], misalignment: [f64; 4]) -> Mat3 {
    let a = angles[0];
    let b = angles[1];
    let u = a.tan();
    let v = b.tan();
    let k = [
        [-u * v / (1. + u * u), 1., -v / (1. + u * u)],
        [-1., u * v / (1. + v * v), u / (1. + v * v)],
    ];
    let kt = [[k[0][0], k[1][0]], [k[0][1], k[1][1]], [k[0][2], k[1][2]]];
    let ci = inverse2(cov);
    let mid = mul32x23(mul32x22(kt, ci), k);
    let mount = kernels::mounting(orientation, misalignment);
    matmul(transpose(mount), matmul(mid, mount))
}

fn direction_jacobian(s: Vec3) -> [[f64; 3]; 2] {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let dx = x * x + z * z;
    let dy = y * y + z * z;
    [[z / dx, 0., -x / dx], [0., z / dy, -y / dy]]
}
fn matmul23(a: [[f64; 3]; 2], b: Mat3) -> [[f64; 3]; 2] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn mul32x22(a: [[f64; 2]; 3], b: Mat2) -> [[f64; 2]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..2).map(|k| a[i][k] * b[k][j]).sum()))
}
fn mul32x23(a: [[f64; 2]; 3], b: [[f64; 3]; 2]) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..2).map(|k| a[i][k] * b[k][j]).sum()))
}
fn inverse2(a: Mat2) -> Mat2 {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]]
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

//! Six-state multiplicative EKF prediction and measurement corrections.

use super::base::{AttitudeEstimator, Mat3, Mat6};

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeMEKF {
    pub base: AttitudeEstimator,
}

#[derive(Clone, Debug, Default)]
pub struct SensorConfig {
    pub code: Vec<i64>,
    pub orientation: Vec<[f64; 4]>,
    pub misalignment: Vec<[f64; 4]>,
    pub bias: Vec<[f64; 6]>,
    pub noise_cholesky: Vec<[[f64; 6]; 6]>,
    pub enabled: Vec<bool>,
    pub offsets: Vec<usize>,
}

impl AttitudeMEKF {
    pub fn new(base: AttitudeEstimator) -> Self {
        Self { base }
    }
}

pub fn predict(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    acceleration_psd: Mat3,
    dt: f64,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    if dt < 0.0 {
        return Err("dt must be nonnegative");
    }
    let rotation = w.map(|x| -x * dt);
    let increment = rotation_to_q(rotation);
    let q_next = normalize(mul(increment, q));
    let mut phi = identity6();
    let rot = dcm(increment);
    for i in 0..3 {
        for j in 0..3 {
            phi[i][j] = rot[i][j];
        }
    }
    let jl = so3_left_jacobian(rotation);
    for i in 0..3 {
        for j in 0..3 {
            phi[i][j + 3] = -jl[i][j] * dt;
        }
    }
    let mut noise = [[0.; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            noise[i][j] = acceleration_psd[i][j] * dt.powi(3) / 3.0;
            noise[i][j + 3] = -acceleration_psd[i][j] * dt.powi(2) / 2.0;
            noise[i + 3][j] = noise[i][j + 3];
            noise[i + 3][j + 3] = acceleration_psd[i][j] * dt;
        }
    }
    let result = add6(matmul6(matmul6(phi, p), transpose6(phi)), noise);
    Ok((q_next, w, sym6(result)))
}

pub fn correct(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    residual: [f64; 3],
    h: [[f64; 6]; 3],
    r: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    let pht = matmul6x3x6(p, h);
    let s = add3m(matmul3x6x6x3(h, pht), r);
    let sinv = inverse3(s).ok_or("measurement innovation covariance is singular")?;
    let k = matmul6x3(pht, sinv);
    let delta = mul6x3v(k, residual);
    let dq = rotation_to_q([delta[0], delta[1], delta[2]]);
    let q_next = normalize(mul(dq, q));
    let w_next = [w[0] + delta[3], w[1] + delta[4], w[2] + delta[5]];
    let kh = matmul6x3x3x6(k, h);
    let mut remainder = identity6();
    for i in 0..6 {
        for j in 0..6 {
            remainder[i][j] -= kh[i][j];
        }
    }
    let joseph = add6(
        matmul6(matmul6(remainder, p), transpose6(remainder)),
        matmul6x3x3x6_cov(k, r),
    );
    let mut reset = identity6();
    let skew = skew([delta[0], delta[1], delta[2]]);
    for i in 0..3 {
        for j in 0..3 {
            reset[i][j] += 0.5 * skew[i][j];
        }
    }
    let updated = sym6(matmul6(matmul6(reset, joseph), transpose6(reset)));
    Ok((q_next, w_next, updated))
}

pub fn correct_gyro(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    measured_rate: [f64; 3],
    mounting: Mat3,
    covariance: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    let predicted = mul3v(mounting, w);
    let residual = [
        measured_rate[0] - predicted[0],
        measured_rate[1] - predicted[1],
        measured_rate[2] - predicted[2],
    ];
    let mut h = [[0.; 6]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j + 3] = mounting[i][j];
        }
    }
    correct(q, w, p, residual, h, covariance)
}

pub fn correct_star_tracker(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    measured: [f64; 4],
    mounting_q: [f64; 4],
    covariance: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    let predicted = mul(mounting_q, q);
    let relative = mul(measured, conjugate(predicted));
    let residual = quaternion_to_rotation_vector(relative);
    let mounting = dcm(mounting_q);
    let mut h = [[0.; 6]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] = mounting[i][j];
        }
    }
    correct(q, w, p, residual, h, covariance)
}

/// Predict and process valid gyro and tracker readings from prepared sensor rows.
pub fn step(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    acceleration_psd: Mat3,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    let (mut q, mut w, mut p) = predict(q, w, p, acceleration_psd, dt)?;
    for i in 0..config.code.len() {
        let kind = config.code[i];
        if !config.enabled[i] || (kind != 0 && kind != 2) {
            continue;
        }
        let offset = config.offsets[i];
        let width = if kind == 0 { 3 } else { 4 };
        let Some(value) = readings.get(offset..offset + width) else {
            continue;
        };
        if value.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let mounting_q = normalize(mul(config.misalignment[i], config.orientation[i]));
        let mut covariance = [[0.; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    covariance[row][col] +=
                        config.noise_cholesky[i][row][k] * config.noise_cholesky[i][col][k];
                }
            }
        }
        if kind == 0 {
            let measured = [
                value[0] - config.bias[i][0],
                value[1] - config.bias[i][1],
                value[2] - config.bias[i][2],
            ];
            (q, w, p) = correct_gyro(q, w, p, measured, dcm(mounting_q), covariance)?;
        } else {
            (q, w, p) = correct_star_tracker(
                q,
                w,
                p,
                [value[0], value[1], value[2], value[3]],
                mounting_q,
                covariance,
            )?;
        }
    }
    Ok((q, w, p))
}

fn rotation_to_q(v: [f64; 3]) -> [f64; 4] {
    let angle = dot3(v, v).sqrt();
    if angle < 1e-12 {
        return normalize([1. - angle * angle / 8., v[0] * 0.5, v[1] * 0.5, v[2] * 0.5]);
    }
    let scale = (angle * 0.5).sin() / angle;
    [
        (angle * 0.5).cos(),
        v[0] * scale,
        v[1] * scale,
        v[2] * scale,
    ]
}
fn quaternion_to_rotation_vector(q: [f64; 4]) -> [f64; 3] {
    let mut q = normalize(q);
    if q[0] < 0. {
        q = q.map(|x| -x)
    }
    let n = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        return [2. * q[1], 2. * q[2], 2. * q[3]];
    }
    let k = 2. * n.atan2(q[0]) / n;
    [q[1] * k, q[2] * k, q[3] * k]
}
fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn dcm(q: [f64; 4]) -> Mat3 {
    let q = normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn skew(v: [f64; 3]) -> Mat3 {
    [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]]
}
fn mul3v(a: Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}
fn so3_left_jacobian(v: [f64; 3]) -> Mat3 {
    let theta = dot3(v, v).sqrt();
    let k = skew(v);
    let k2 = matmul3(k, k);
    let (a, b) = if theta < 1e-5 {
        (0.5 - theta * theta / 24., 1. / 6. - theta * theta / 120.)
    } else {
        (
            (1. - theta.cos()) / theta.powi(2),
            (theta - theta.sin()) / theta.powi(3),
        )
    };
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (if i == j { 1. } else { 0. }) + a * k[i][j] + b * k2[i][j])
    })
}
fn identity6() -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}
fn matmul6(a: Mat6, b: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[k][j]).sum()))
}
fn transpose6(a: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn add6(a: Mat6, b: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn sym6(a: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (a[i][j] + a[j][i])))
}
fn matmul3(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn matmul6x3x6(a: Mat6, b: [[f64; 6]; 3]) -> [[f64; 3]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[j][k]).sum()))
}
fn matmul3x6x6x3(a: [[f64; 6]; 3], b: [[f64; 3]; 6]) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[k][j]).sum()))
}
fn add3m(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn matmul6x3(a: [[f64; 3]; 6], b: Mat3) -> [[f64; 3]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn mul6x3v(a: [[f64; 3]; 6], v: [f64; 3]) -> [f64; 6] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}
fn matmul6x3x3x6(a: [[f64; 3]; 6], b: [[f64; 6]; 3]) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn matmul6x3x3x6_cov(k: [[f64; 3]; 6], r: Mat3) -> Mat6 {
    matmul6x3x3x6(matmul6x3(k, r), transpose6x3(k))
}
fn transpose6x3(a: [[f64; 3]; 6]) -> [[f64; 6]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn inverse3(a: Mat3) -> Option<Mat3> {
    let d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if d.abs() < 1e-30 {
        return None;
    }
    Some(
        [
            [
                a[1][1] * a[2][2] - a[1][2] * a[2][1],
                a[0][2] * a[2][1] - a[0][1] * a[2][2],
                a[0][1] * a[1][2] - a[0][2] * a[1][1],
            ],
            [
                a[1][2] * a[2][0] - a[1][0] * a[2][2],
                a[0][0] * a[2][2] - a[0][2] * a[2][0],
                a[0][2] * a[1][0] - a[0][0] * a[1][2],
            ],
            [
                a[1][0] * a[2][1] - a[1][1] * a[2][0],
                a[0][1] * a[2][0] - a[0][0] * a[2][1],
                a[0][0] * a[1][1] - a[0][1] * a[1][0],
            ],
        ]
        .map(|row| row.map(|x| x / d)),
    )
}

//! Shared magnetic and optical direction-sensor corrections for 6-state MEKF and 7-state EKF.

use std::f64::consts::PI;

use super::attitude_mekf::SensorConfig;
use super::base::Mat3;

/// Result of a single direction-sensor measurement update (width 1, 2, or 3).
#[derive(Clone, Debug)]
pub struct DirectionCorrection<const N: usize> {
    pub quaternion: [f64; 4],
    pub rate: [f64; 3],
    pub covariance: [[f64; N]; N],
    pub innovation: [f64; 3],
    pub innovation_covariance: Mat3,
    pub gain: [[f64; 3]; N],
    pub residual: [f64; 3],
    pub width: usize,
}

/// Predict the measurement of direction sensor `index` (codes 1, 3, 4, 5) at attitude `q`.
///
/// Returns a 3-element array with valid entries in `0..width` (`width` = 3 for code 1,
/// 1 for code 3, 2 for codes 4 and 5). Unavailable fine sensors return `f64::NAN`.
pub fn direction_prediction(
    config: &SensorConfig,
    index: usize,
    q: [f64; 4],
) -> Result<(usize, [f64; 3]), &'static str> {
    let code = config.code[index];
    let reference = config
        .reference
        .get(index)
        .copied()
        .ok_or("magnetic and optical corrections require inertial references")?;
    if !reference.iter().all(|x| x.is_finite()) {
        return Err("magnetic and optical corrections require finite inertial references");
    }

    let q_norm = normalize(q);
    let mount_q = normalize(mul(config.misalignment[index], config.orientation[index]));
    let direction = mul3v(dcm(mount_q), mul3v(dcm(q_norm), reference));
    let bias = config.bias[index];
    let scalars = config
        .scalars
        .get(index)
        .copied()
        .unwrap_or([0.0, 0.0, 0.0]);

    match code {
        1 => {
            let sat = scalars[0];
            Ok((
                3,
                [
                    (direction[0] + bias[0]).max(-sat).min(sat),
                    (direction[1] + bias[1]).max(-sat).min(sat),
                    (direction[2] + bias[2]).max(-sat).min(sat),
                ],
            ))
        }
        3 => {
            let sat = scalars[0];
            let sensitivity = scalars[1];
            let normal = config
                .normal
                .get(index)
                .copied()
                .unwrap_or([0.0, 0.0, 1.0]);
            let val = (sensitivity * dot3(normal, direction).max(0.0) + bias[0])
                .max(0.0)
                .min(sat);
            Ok((1, [val, 0.0, 0.0]))
        }
        4 | 5 => {
            let angle_x = direction[0].atan2(direction[2]);
            let angle_y = direction[1].atan2(direction[2]);
            let fov = config.fov.get(index).copied().unwrap_or([0.0, 0.0]);
            if angle_x.abs() > fov[0] || angle_y.abs() > fov[1] {
                return Ok((2, [f64::NAN, f64::NAN, 0.0]));
            }
            Ok((2, [angle_x + bias[0], angle_y + bias[1], 0.0]))
        }
        _ => Err("unsupported direction sensor code"),
    }
}

/// Apply a direction-sensor measurement update to a 6-state MEKF or 7-state EKF state.
///
/// Returns `Ok(None)` when the predicted measurement is unavailable (outside FOV) or when
/// the finite-difference Jacobian is non-finite.
pub fn correct_direction<const N: usize>(
    q: [f64; 4],
    w: [f64; 3],
    p: [[f64; N]; N],
    config: &SensorConfig,
    index: usize,
    observed: &[f64],
) -> Result<Option<DirectionCorrection<N>>, &'static str> {
    assert!(N == 6 || N == 7, "state dimension must be 6 or 7");
    let full = N == 7;
    let code = config.code[index];
    let (width, expected) = direction_prediction(config, index, q)?;
    if expected[..width].iter().any(|x| !x.is_finite()) {
        return Ok(None);
    }
    if observed.len() < width {
        return Err("sensor reading offset is out of range");
    }

    let mut h = [[0.0; N]; 3];
    let eps = 1e-7;
    let cols = if full { 4 } else { 3 };
    for j in 0..cols {
        let (qp, qm) = if full {
            let mut plus = q;
            let mut minus = q;
            plus[j] += eps;
            minus[j] -= eps;
            (normalize(plus), normalize(minus))
        } else {
            let mut d = [0.0; 3];
            d[j] = eps;
            let dq_p = rotation_to_q(d);
            let dq_m = rotation_to_q([-d[0], -d[1], -d[2]]);
            (normalize(mul(dq_p, q)), normalize(mul(dq_m, q)))
        };
        let (_, pred_p) = direction_prediction(config, index, qp)?;
        let (_, pred_m) = direction_prediction(config, index, qm)?;
        for row in 0..width {
            let deriv = (pred_p[row] - pred_m[row]) / (2.0 * eps);
            if !deriv.is_finite() {
                return Ok(None);
            }
            h[row][j] = deriv;
        }
    }

    let mut residual = [f64::NAN; 3];
    for i in 0..width {
        let mut r = observed[i] - expected[i];
        if code == 4 || code == 5 {
            r = (r + PI).rem_euclid(2.0 * PI) - PI;
        }
        residual[i] = r;
    }

    let mut r_mat = [[0.0; 3]; 3];
    if code == 3 {
        let noise_std = config.scalars.get(index).map(|s| s[2]).unwrap_or(0.0);
        r_mat[0][0] = noise_std * noise_std;
    } else {
        let chol = &config.noise_cholesky[index];
        for a in 0..width {
            for b in 0..width {
                r_mat[a][b] = (0..width).map(|k| chol[a][k] * chol[b][k]).sum();
            }
        }
    }

    // P * H^T (N x width)
    let mut pht = [[0.0; 3]; N];
    for i in 0..N {
        for j in 0..width {
            pht[i][j] = (0..N).map(|k| p[i][k] * h[j][k]).sum();
        }
    }

    // S = H * P * H^T + R (width x width)
    let mut s = [[f64::NAN; 3]; 3];
    for i in 0..width {
        for j in 0..width {
            s[i][j] = (0..N).map(|k| h[i][k] * pht[k][j]).sum::<f64>() + r_mat[i][j];
        }
    }

    let sinv = inverse_submatrix(s, width).ok_or("direction innovation covariance is singular")?;

    // Gain K = P * H^T * S^-1 (N x width)
    let mut gain = [[f64::NAN; 3]; N];
    for i in 0..N {
        for j in 0..width {
            gain[i][j] = (0..width).map(|k| pht[i][k] * sinv[k][j]).sum();
        }
    }

    let mut delta = [0.0; N];
    for i in 0..N {
        delta[i] = (0..width).map(|j| gain[i][j] * residual[j]).sum();
    }

    // A = I - K * H (N x N)
    let mut a_mat = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let kh: f64 = (0..width).map(|k| gain[i][k] * h[k][j]).sum();
            a_mat[i][j] = (if i == j { 1.0 } else { 0.0 }) - kh;
        }
    }

    // Joseph form: P_next = A * P * A^T + K * R * K^T
    let mut ap = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            ap[i][j] = (0..N).map(|k| a_mat[i][k] * p[k][j]).sum();
        }
    }
    let mut kr = [[0.0; 3]; N];
    for i in 0..N {
        for j in 0..width {
            kr[i][j] = (0..width).map(|k| gain[i][k] * r_mat[k][j]).sum();
        }
    }
    let mut joseph = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let ap_at: f64 = (0..N).map(|k| ap[i][k] * a_mat[j][k]).sum();
            let kr_kt: f64 = (0..width).map(|k| kr[i][k] * gain[j][k]).sum();
            joseph[i][j] = ap_at + kr_kt;
        }
    }

    let (q_next, w_next, reset) = if full {
        let mut qn = [
            q[0] + delta[0],
            q[1] + delta[1],
            q[2] + delta[2],
            q[3] + delta[3],
        ];
        let qn_norm = qn.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !qn.iter().all(|x| x.is_finite()) || qn_norm == 0.0 {
            return Err("EKF quaternion correction is invalid");
        }
        qn = qn.map(|x| x / qn_norm);
        let wn = [w[0] + delta[4], w[1] + delta[5], w[2] + delta[6]];
        let mut res = [[0.0; N]; N];
        for i in 0..N {
            res[i][i] = 1.0;
        }
        for i in 0..4 {
            for j in 0..4 {
                res[i][j] -= qn[i] * qn[j];
            }
        }
        (qn, wn, res)
    } else {
        let d = [delta[0], delta[1], delta[2]];
        let dq = rotation_to_q(d);
        let qn = normalize(mul(dq, q));
        let wn = [w[0] + delta[3], w[1] + delta[4], w[2] + delta[5]];
        let mut res = [[0.0; N]; N];
        for i in 0..N {
            res[i][i] = 1.0;
        }
        let sk = skew(d);
        for i in 0..3 {
            for j in 0..3 {
                res[i][j] += 0.5 * sk[i][j];
            }
        }
        (qn, wn, res)
    };

    // P_reset = reset * joseph * reset^T, symmetrized
    let mut rj = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            rj[i][j] = (0..N).map(|k| reset[i][k] * joseph[k][j]).sum();
        }
    }
    let mut p_next = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            p_next[i][j] = (0..N).map(|k| rj[i][k] * reset[j][k]).sum();
        }
    }
    let mut p_sym = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            p_sym[i][j] = 0.5 * (p_next[i][j] + p_next[j][i]);
        }
    }

    let mut post_residual = [f64::NAN; 3];
    for i in 0..width {
        let h_delta: f64 = (0..N).map(|j| h[i][j] * delta[j]).sum();
        post_residual[i] = residual[i] - h_delta;
    }

    Ok(Some(DirectionCorrection {
        quaternion: q_next,
        rate: w_next,
        covariance: p_sym,
        innovation: residual,
        innovation_covariance: s,
        gain,
        residual: post_residual,
        width,
    }))
}

pub(crate) fn inverse_submatrix(a: Mat3, width: usize) -> Option<Mat3> {
    let mut out = [[0.0; 3]; 3];
    match width {
        1 => {
            if a[0][0].abs() < 1e-60 || !a[0][0].is_finite() {
                return None;
            }
            out[0][0] = 1.0 / a[0][0];
            Some(out)
        }
        2 => {
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            if det.abs() < 1e-60 || !det.is_finite() {
                return None;
            }
            out[0][0] = a[1][1] / det;
            out[0][1] = -a[0][1] / det;
            out[1][0] = -a[1][0] / det;
            out[1][1] = a[0][0] / det;
            Some(out)
        }
        3 => {
            let d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            if d.abs() < 1e-60 || !d.is_finite() {
                return None;
            }
            out = [
                [
                    (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d,
                    (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d,
                    (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d,
                ],
                [
                    (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d,
                    (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d,
                    (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d,
                ],
                [
                    (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d,
                    (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d,
                    (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d,
                ],
            ];
            Some(out)
        }
        _ => None,
    }
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

fn rotation_to_q(v: [f64; 3]) -> [f64; 4] {
    let angle = dot3(v, v).sqrt();
    if angle < 1e-12 {
        return normalize([1.0 - angle * angle / 8.0, v[0] * 0.5, v[1] * 0.5, v[2] * 0.5]);
    }
    let scale = (angle * 0.5).sin() / angle;
    [
        (angle * 0.5).cos(),
        v[0] * scale,
        v[1] * scale,
        v[2] * scale,
    ]
}

fn dcm(q: [f64; 4]) -> Mat3 {
    let q = normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mul3v(a: Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}

fn skew(v: [f64; 3]) -> Mat3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

//! Seven-state quaternion EKF prediction and measurement corrections.

use super::attitude_mekf::SensorConfig;
use super::base::{close, min_eigenvalue, min_eigenvalue3, symmetric3, Mat3, DIAGNOSTIC_FIELDS};
use super::direction_correction::{correct_direction, inverse_submatrix};

pub type Mat7 = [[f64; 7]; 7];

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeEKF {
    pub quaternion: [f64; 4],
    pub rate: [f64; 3],
    pub covariance: Mat7,
    pub acceleration_psd: Mat3,
    pub record: Vec<String>,
}

impl AttitudeEKF {
    pub fn new(
        quaternion: [f64; 4],
        rate: [f64; 3],
        covariance: Mat7,
        acceleration_psd: Mat3,
        record: Vec<String>,
    ) -> Result<Self, &'static str> {
        if !quaternion.iter().all(|x| x.is_finite()) {
            return Err("quaternion must be a finite nonzero Hamilton 4-vector");
        }
        let qnorm = quaternion.iter().map(|x| x * x).sum::<f64>().sqrt();
        if qnorm == 0.0 {
            return Err("quaternion must be a finite nonzero Hamilton 4-vector");
        }
        if !rate.iter().all(|x| x.is_finite()) {
            return Err("rate must be a finite 3-vector");
        }
        if !covariance.iter().flatten().all(|x| x.is_finite()) || !symmetric7(covariance) {
            return Err("covariance must be a finite symmetric 7x7 matrix");
        }
        if min_eigenvalue7(covariance) < -1e-12 {
            return Err("covariance must be positive semidefinite");
        }
        if !acceleration_psd.iter().flatten().all(|x| x.is_finite())
            || !symmetric3(acceleration_psd)
        {
            return Err("acceleration_psd must be a finite symmetric 3x3 matrix");
        }
        if min_eigenvalue3(acceleration_psd) < -1e-12 {
            return Err("acceleration_psd must be positive semidefinite");
        }
        if record
            .iter()
            .any(|name| !DIAGNOSTIC_FIELDS.contains(&name.as_str()))
        {
            return Err("unknown estimator diagnostic field");
        }
        Ok(Self {
            quaternion: quaternion.map(|x| x / qnorm),
            rate,
            covariance,
            acceleration_psd,
            record,
        })
    }

    pub fn prepare(&self) -> ([f64; 4], [f64; 3], Mat7, Mat3) {
        (
            self.quaternion,
            self.rate,
            self.covariance,
            self.acceleration_psd,
        )
    }
}

pub fn predict(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    acceleration_psd: Mat3,
    dt: f64,
) -> Result<([f64; 4], [f64; 3], Mat7), &'static str> {
    if !dt.is_finite() || dt < 0.0 {
        return Err("dt must be finite and nonnegative");
    }
    let mut x = [0.0; 7];
    x[..4].copy_from_slice(&q);
    x[4..].copy_from_slice(&w);

    let mut xp = transition(x, dt);
    if dot4([xp[0], xp[1], xp[2], xp[3]], q) < 0.0 {
        for i in 0..4 {
            xp[i] = -xp[i];
        }
    }
    let q_next = [xp[0], xp[1], xp[2], xp[3]];
    let w_next = [xp[4], xp[5], xp[6]];

    let mut f = [[0.0; 7]; 7];
    let eps = 1e-7;
    for col in 0..7 {
        let mut plus = x;
        let mut minus = x;
        plus[col] += eps;
        minus[col] -= eps;
        let mut yp = transition(plus, dt);
        let mut ym = transition(minus, dt);
        if dot4([yp[0], yp[1], yp[2], yp[3]], q_next) < 0.0 {
            for i in 0..4 {
                yp[i] = -yp[i];
            }
        }
        if dot4([ym[0], ym[1], ym[2], ym[3]], q_next) < 0.0 {
            for i in 0..4 {
                ym[i] = -ym[i];
            }
        }
        for row in 0..7 {
            f[row][col] = (yp[row] - ym[row]) / (2.0 * eps);
        }
    }

    let mut q6 = [[0.0; 6]; 6];
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    for i in 0..3 {
        for j in 0..3 {
            q6[i][j] = acceleration_psd[i][j] * (dt3 / 3.0);
            q6[i][j + 3] = -acceleration_psd[i][j] * (dt2 / 2.0);
            q6[i + 3][j] = q6[i][j + 3];
            q6[i + 3][j + 3] = acceleration_psd[i][j] * dt;
        }
    }

    let mut lift = [[0.0; 6]; 7];
    for j in 0..3 {
        lift[0][j] = -0.5 * q_next[j + 1];
    }
    let sk = skew([q_next[1], q_next[2], q_next[3]]);
    for i in 0..3 {
        for j in 0..3 {
            lift[i + 1][j] = 0.5 * ((if i == j { q_next[0] } else { 0.0 }) - sk[i][j]);
        }
    }
    for i in 0..3 {
        lift[i + 4][i + 3] = 1.0;
    }

    // F * P * F^T + lift * q6 * lift^T
    let fpf = matmul7(matmul7(f, p), transpose7(f));
    let mut lq = [[0.0; 6]; 7];
    for i in 0..7 {
        for j in 0..6 {
            lq[i][j] = (0..6).map(|k| lift[i][k] * q6[k][j]).sum();
        }
    }
    let mut lql = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            lql[i][j] = (0..6).map(|k| lq[i][k] * lift[j][k]).sum();
        }
    }
    let combined = add7(fpf, lql);
    let p_next = tangent_covariance(q_next, combined);
    Ok((q_next, w_next, p_next))
}

struct EkfCorrection {
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    s: Mat3,
    gain: [[f64; 3]; 7],
    post_residual: [f64; 3],
}

fn correct_detailed(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    residual: [f64; 3],
    h: [[f64; 7]; 3],
    r: Mat3,
) -> Result<EkfCorrection, &'static str> {
    // P * H^T (7 x 3)
    let mut pht = [[0.0; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            pht[i][j] = (0..7).map(|k| p[i][k] * h[j][k]).sum();
        }
    }
    // S = H * P * H^T + R (3 x 3)
    let mut s = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = (0..7).map(|k| h[i][k] * pht[k][j]).sum::<f64>() + r[i][j];
        }
    }
    let sinv = inverse_submatrix(s, 3).ok_or("innovation covariance is singular")?;
    // K = P * H^T * S^-1 (7 x 3)
    let mut gain = [[0.0; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            gain[i][j] = (0..3).map(|k| pht[i][k] * sinv[k][j]).sum();
        }
    }
    let mut delta = [0.0; 7];
    for i in 0..7 {
        delta[i] = (0..3).map(|j| gain[i][j] * residual[j]).sum();
    }
    let mut q_next = [
        q[0] + delta[0],
        q[1] + delta[1],
        q[2] + delta[2],
        q[3] + delta[3],
    ];
    let qnorm = q_next.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !q_next.iter().all(|x| x.is_finite()) || qnorm == 0.0 {
        return Err("EKF quaternion correction is invalid");
    }
    q_next = q_next.map(|x| x / qnorm);
    let w_next = [w[0] + delta[4], w[1] + delta[5], w[2] + delta[6]];

    let mut remainder = identity7();
    for i in 0..7 {
        for j in 0..7 {
            remainder[i][j] -= (0..3).map(|k| gain[i][k] * h[k][j]).sum::<f64>();
        }
    }
    let joseph_left = matmul7(matmul7(remainder, p), transpose7(remainder));
    let mut kr = [[0.0; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            kr[i][j] = (0..3).map(|k| gain[i][k] * r[k][j]).sum();
        }
    }
    let mut krk = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            krk[i][j] = (0..3).map(|k| kr[i][k] * gain[j][k]).sum();
        }
    }
    let p_next = tangent_covariance(q_next, add7(joseph_left, krk));

    let post_residual = std::array::from_fn(|i| {
        residual[i] - (0..7).map(|j| h[i][j] * delta[j]).sum::<f64>()
    });

    Ok(EkfCorrection {
        q: q_next,
        w: w_next,
        p: p_next,
        s,
        gain,
        post_residual,
    })
}

pub fn correct(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    residual: [f64; 3],
    h: [[f64; 7]; 3],
    r: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat7), &'static str> {
    let c = correct_detailed(q, w, p, residual, h, r)?;
    Ok((c.q, c.w, c.p))
}

pub fn correct_gyro(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    measured_rate: [f64; 3],
    mounting: Mat3,
    covariance: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat7), &'static str> {
    let predicted = mul3v(mounting, w);
    let residual = [
        measured_rate[0] - predicted[0],
        measured_rate[1] - predicted[1],
        measured_rate[2] - predicted[2],
    ];
    let mut h = [[0.0; 7]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j + 4] = mounting[i][j];
        }
    }
    correct(q, w, p, residual, h, covariance)
}

pub fn correct_star_tracker(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    measured: [f64; 4],
    mounting_q: [f64; 4],
    covariance: Mat3,
) -> Result<([f64; 4], [f64; 3], Mat7), &'static str> {
    let mount = normalize(mounting_q);
    let residual = tracker_residual(q, measured, mount);
    let mut h = [[0.0; 7]; 3];
    let eps = 1e-7;
    for col in 0..4 {
        let mut plus = q;
        let mut minus = q;
        plus[col] += eps;
        minus[col] -= eps;
        plus = normalize(plus);
        minus = normalize(minus);
        let rp = tracker_residual(plus, measured, mount);
        let rm = tracker_residual(minus, measured, mount);
        for row in 0..3 {
            h[row][col] = -(rp[row] - rm[row]) / (2.0 * eps);
        }
    }
    correct(q, w, p, residual, h, covariance)
}

/// Predict and process valid attitude sensor readings from prepared sensor rows.
pub fn step(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat7,
    acceleration_psd: Mat3,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
) -> Result<([f64; 4], [f64; 3], Mat7), &'static str> {
    let (mut q, mut w, mut p) = predict(q, w, p, acceleration_psd, dt)?;
    for i in 0..config.code.len() {
        let kind = config.code[i];
        if !config.enabled[i] || !(0..=5).contains(&kind) {
            continue;
        }
        let width = match kind {
            0 | 1 => 3,
            2 => 4,
            3 => 1,
            _ => 2,
        };
        let offset = config.offsets[i];
        let Some(value) = readings.get(offset..offset + width) else {
            return Err("sensor reading offset is out of range");
        };
        if value.iter().any(|x| !x.is_finite()) {
            continue;
        }
        if matches!(kind, 1 | 3 | 4 | 5) {
            if let Some(corr) = correct_direction::<7>(q, w, p, config, i, value)? {
                q = corr.quaternion;
                w = corr.rate;
                p = corr.covariance;
            }
            continue;
        }
        let mounting_q = normalize(mul(config.misalignment[i], config.orientation[i]));
        let mut covariance = [[0.0; 3]; 3];
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

/// One subsystem tick for EKF, including diagnostics of the last valid correction.
/// Layout:
/// - `0..4`: quaternion
/// - `4..7`: rate
/// - `7..56`: 7x7 covariance (row-major)
/// - `56..59`: measurement_innovation
/// - `59..68`: measurement_innovation_covariance (3x3)
/// - `68`: normalized_innovation_squared
/// - `69`: normalized_estimation_error_squared
/// - `70..73`: measurement_residual
/// - `73..91`: kalman_gain (6x3 tangent-space gain)
/// - `91..100`: estimated_attitude_covariance (3x3 tangent-space)
/// - `100..109`: estimated_rate_covariance (3x3)
pub fn step_diagnostics(
    state: &mut AttitudeEKF,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
    truth: &[f64],
    nees: bool,
) -> Result<[f64; 109], &'static str> {
    let (mut q, mut w, mut p) = predict(
        state.quaternion,
        state.rate,
        state.covariance,
        state.acceleration_psd,
        dt,
    )?;
    let mut out = [f64::NAN; 109];
    for i in 0..config.code.len() {
        let kind = config.code[i];
        if !config.enabled[i] || !(0..=5).contains(&kind) {
            continue;
        }
        let width = match kind {
            0 | 1 => 3,
            2 => 4,
            3 => 1,
            _ => 2,
        };
        let offset = config.offsets[i];
        let value = readings
            .get(offset..offset + width)
            .ok_or("invalid sensor offset")?;
        if value.iter().any(|x| !x.is_finite()) {
            continue;
        }
        if matches!(kind, 1 | 3 | 4 | 5) {
            let Some(corr) = correct_direction::<7>(q, w, p, config, i, value)? else {
                continue;
            };
            let m = corr.width;
            for a in 0..3 {
                out[56 + a] = if a < m { corr.innovation[a] } else { f64::NAN };
                out[70 + a] = if a < m { corr.residual[a] } else { f64::NAN };
                for b in 0..3 {
                    out[59 + a * 3 + b] = if a < m && b < m {
                        corr.innovation_covariance[a][b]
                    } else {
                        f64::NAN
                    };
                }
            }
            let sinv = inverse_submatrix(corr.innovation_covariance, m)
                .ok_or("direction innovation covariance is singular")?;
            let mut nis = 0.0;
            for a in 0..m {
                for b in 0..m {
                    nis += corr.innovation[a] * sinv[a][b] * corr.innovation[b];
                }
            }
            out[68] = nis;
            let err_j = state_error_jacobian(q);
            for a in 0..6 {
                for b in 0..3 {
                    out[73 + a * 3 + b] = if b < m {
                        (0..7).map(|k| err_j[a][k] * corr.gain[k][b]).sum()
                    } else {
                        f64::NAN
                    };
                }
            }
            q = corr.quaternion;
            w = corr.rate;
            p = corr.covariance;
            continue;
        }
        let mount = normalize(mul(config.misalignment[i], config.orientation[i]));
        let mut r = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                r[a][b] = (0..3)
                    .map(|k| config.noise_cholesky[i][a][k] * config.noise_cholesky[i][b][k])
                    .sum();
            }
        }
        let mut h = [[0.0; 7]; 3];
        let residual = if kind == 0 {
            let rot = dcm(mount);
            for a in 0..3 {
                for b in 0..3 {
                    h[a][b + 4] = rot[a][b];
                }
            }
            let predicted = mul3v(rot, w);
            std::array::from_fn(|j| value[j] - config.bias[i][j] - predicted[j])
        } else {
            let meas = [value[0], value[1], value[2], value[3]];
            let eps = 1e-7;
            for col in 0..4 {
                let mut plus = q;
                let mut minus = q;
                plus[col] += eps;
                minus[col] -= eps;
                plus = normalize(plus);
                minus = normalize(minus);
                let rp = tracker_residual(plus, meas, mount);
                let rm = tracker_residual(minus, meas, mount);
                for row in 0..3 {
                    h[row][col] = -(rp[row] - rm[row]) / (2.0 * eps);
                }
            }
            tracker_residual(q, meas, mount)
        };
        let corr = correct_detailed(q, w, p, residual, h, r)?;
        out[56..59].copy_from_slice(&residual);
        for a in 0..3 {
            for b in 0..3 {
                out[59 + a * 3 + b] = corr.s[a][b];
            }
        }
        let sinv = inverse_submatrix(corr.s, 3).ok_or("innovation covariance is singular")?;
        out[68] = dot3(residual, mul3v(sinv, residual));
        out[70..73].copy_from_slice(&corr.post_residual);
        let err_j = state_error_jacobian(q);
        for a in 0..6 {
            for b in 0..3 {
                out[73 + a * 3 + b] = (0..7).map(|k| err_j[a][k] * corr.gain[k][b]).sum();
            }
        }
        q = corr.q;
        w = corr.w;
        p = corr.p;
    }

    out[..4].copy_from_slice(&q);
    out[4..7].copy_from_slice(&w);
    for a in 0..7 {
        for b in 0..7 {
            out[7 + a * 7 + b] = p[a][b];
        }
    }
    let err_j = state_error_jacobian(q);
    let mut err_cov = [[0.0; 6]; 6];
    let mut jp = [[0.0; 7]; 6];
    for a in 0..6 {
        for b in 0..7 {
            jp[a][b] = (0..7).map(|k| err_j[a][k] * p[k][b]).sum();
        }
    }
    for a in 0..6 {
        for b in 0..6 {
            err_cov[a][b] = (0..7).map(|k| jp[a][k] * err_j[b][k]).sum();
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            out[91 + a * 3 + b] = err_cov[a][b];
            out[100 + a * 3 + b] = err_cov[a + 3][b + 3];
        }
    }
    if nees {
        let rotation = quaternion_to_rotation_vector(mul(
            [truth[0], truth[1], truth[2], truth[3]],
            conjugate(q),
        ));
        let error = [
            rotation[0],
            rotation[1],
            rotation[2],
            truth[4] - w[0],
            truth[5] - w[1],
            truth[6] - w[2],
        ];
        let mut l = [[0.0; 6]; 6];
        let mut z = [0.0; 6];
        for a in 0..6 {
            for b in 0..=a {
                let v = err_cov[a][b] - (0..b).map(|k| l[a][k] * l[b][k]).sum::<f64>();
                if a == b {
                    if v <= 0.0 {
                        return Err("EKF tangent covariance is singular");
                    }
                    l[a][b] = v.sqrt();
                } else {
                    l[a][b] = v / l[b][b];
                }
            }
            z[a] = (error[a] - (0..a).map(|b| l[a][b] * z[b]).sum::<f64>()) / l[a][a];
        }
        out[69] = z.iter().map(|x| x * x).sum();
    }
    state.quaternion = q;
    state.rate = w;
    state.covariance = p;
    Ok(out)
}

fn tracker_residual(q: [f64; 4], measured: [f64; 4], mount: [f64; 4]) -> [f64; 3] {
    let predicted = mul(mount, q);
    let obs = if dot4(measured, predicted) < 0.0 {
        measured.map(|x| -x)
    } else {
        measured
    };
    quaternion_to_rotation_vector(mul(obs, conjugate(predicted)))
}

fn transition(x: [f64; 7], dt: f64) -> [f64; 7] {
    let q = [x[0], x[1], x[2], x[3]];
    let w = [x[4], x[5], x[6]];
    let q_next = normalize(mul(rotation_to_q([-w[0] * dt, -w[1] * dt, -w[2] * dt]), q));
    [q_next[0], q_next[1], q_next[2], q_next[3], w[0], w[1], w[2]]
}

fn state_error_jacobian(q: [f64; 4]) -> [[f64; 7]; 6] {
    let mut j = [[0.0; 7]; 6];
    for i in 0..3 {
        j[i][0] = -2.0 * q[i + 1];
    }
    let sk = skew([q[1], q[2], q[3]]);
    for a in 0..3 {
        for b in 0..3 {
            j[a][b + 1] = 2.0 * ((if a == b { q[0] } else { 0.0 }) + sk[a][b]);
        }
    }
    for i in 0..3 {
        j[i + 3][i + 4] = 1.0;
    }
    j
}

fn tangent_covariance(q: [f64; 4], p: Mat7) -> Mat7 {
    let mut proj = identity7();
    for i in 0..4 {
        for j in 0..4 {
            proj[i][j] -= q[i] * q[j];
        }
    }
    sym7(matmul7(matmul7(proj, p), transpose7(proj)))
}

fn symmetric7(a: Mat7) -> bool {
    (0..7).all(|i| (0..7).all(|j| close(a[i][j], a[j][i])))
}

fn min_eigenvalue7(a: Mat7) -> f64 {
    min_eigenvalue(a.iter().map(|r| r.to_vec()).collect())
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

fn quaternion_to_rotation_vector(q: [f64; 4]) -> [f64; 3] {
    let mut q = normalize(q);
    if q[0] < 0.0 {
        q = q.map(|x| -x);
    }
    let n = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        return [2.0 * q[1], 2.0 * q[2], 2.0 * q[3]];
    }
    let k = 2.0 * n.atan2(q[0]) / n;
    [q[1] * k, q[2] * k, q[3] * k]
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

fn skew(v: [f64; 3]) -> Mat3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn mul3v(a: Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}

fn identity7() -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
}

fn matmul7(a: Mat7, b: Mat7) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..7).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose7(a: Mat7) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn add7(a: Mat7, b: Mat7) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}

fn sym7(a: Mat7) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (a[i][j] + a[j][i])))
}

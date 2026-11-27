//! Six-error-state batched unscented Kalman filter (UKF) with stack-allocated arrays.

use super::attitude_mekf::SensorConfig;
use super::attitude_sequential_ukf::{
    cholesky6, conjugate, dcm, identity6, matmul6, mul, mul3v, normalize,
    quaternion_to_rotation_vector, rotation_to_q, sensor_noise_matrix, sigma_points, skew,
    state_deviations, sym6, transpose6, validate_ukf_params, wrap_angle, SIGMA_COUNT,
};
pub use super::attitude_sequential_ukf::{predict, quaternion_mean};
use super::base::{AttitudeEstimator, Mat3, Mat6};
use super::direction_correction::direction_prediction;

pub const COVARIANCE_SIZE: usize = 6;
const MAX_BATCH_MEAS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeBatchedUKF {
    pub base: AttitudeEstimator,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl AttitudeBatchedUKF {
    pub const COVARIANCE_SIZE: usize = COVARIANCE_SIZE;

    pub fn new(
        base: AttitudeEstimator,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Result<Self, &'static str> {
        validate_ukf_params(alpha, beta, kappa)?;
        Ok(Self {
            base,
            alpha,
            beta,
            kappa,
        })
    }

    pub fn ukf_params(&self) -> [f64; 3] {
        [self.alpha, self.beta, self.kappa]
    }
}

fn invert_submatrix<const N: usize>(mat: &[[f64; N]; N], n: usize) -> Option<[[f64; N]; N]> {
    let mut a = [[0.0; N]; N];
    let mut inv = [[0.0; N]; N];
    for i in 0..n {
        inv[i][i] = 1.0;
        for j in 0..n {
            a[i][j] = mat[i][j];
        }
    }
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().take(n).skip(col + 1) {
            let val = a_row[col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }
        let pivot = a[pivot_row][col];
        if !pivot.is_finite() || pivot.abs() < 1e-30 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }
        let inv_pivot = 1.0 / a[col][col];
        for j in 0..n {
            a[col][j] *= inv_pivot;
            inv[col][j] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor != 0.0 {
                for j in 0..n {
                    a[row][j] -= factor * a[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

#[allow(clippy::too_many_arguments)]
fn step_impl(
    q_init: [f64; 4],
    w_init: [f64; 3],
    p_init: Mat6,
    acceleration_psd: Mat3,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
    alpha: f64,
    beta: f64,
    kappa: f64,
    mut diagnostics: Option<&mut [f64; 96]>,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    let (mut q, mut w, mut p) =
        predict(q_init, w_init, p_init, acceleration_psd, dt, alpha, beta, kappa)?;

    let (q_pts, w_pts, w_m, w_c) = sigma_points(q, w, p, alpha, beta, kappa)?;
    let dx_pts = state_deviations(&q_pts, &w_pts, q, w);

    let mut dz_t = [[0.0; SIGMA_COUNT]; MAX_BATCH_MEAS];
    let mut res_all = [0.0; MAX_BATCH_MEAS];
    let mut p_zz = [[0.0; MAX_BATCH_MEAS]; MAX_BATCH_MEAS];
    let mut m_total = 0usize;

    for i in 0..config.code.len() {
        let kind = config.code[i];
        if !config.enabled[i] || !(0..=5).contains(&kind) {
            continue;
        }
        let obs_width = match kind {
            0 | 1 => 3,
            2 => 4,
            3 => 1,
            _ => 2,
        };
        let offset = config.offsets[i];
        let Some(observed) = readings.get(offset..offset + obs_width) else {
            continue;
        };
        if observed.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let meas_width = if matches!(kind, 0 | 1 | 2) { 3 } else { obs_width };
        if m_total + meas_width > MAX_BATCH_MEAS {
            return Err("too many active attitude measurements for batched UKF stack buffer");
        }

        if kind == 0 {
            let mount_q = normalize(mul(config.misalignment[i], config.orientation[i]));
            let mount = dcm(mount_q);
            let mut z_pts = [[0.0; 3]; SIGMA_COUNT];
            let mut z_mean = [0.0; 3];
            for k in 0..SIGMA_COUNT {
                let rot_w = mul3v(mount, w_pts[k]);
                z_pts[k] = [
                    rot_w[0] + config.bias[i][0],
                    rot_w[1] + config.bias[i][1],
                    rot_w[2] + config.bias[i][2],
                ];
                z_mean[0] += w_m[k] * z_pts[k][0];
                z_mean[1] += w_m[k] * z_pts[k][1];
                z_mean[2] += w_m[k] * z_pts[k][2];
            }
            for a in 0..3 {
                for k in 0..SIGMA_COUNT {
                    dz_t[m_total + a][k] = z_pts[k][a] - z_mean[a];
                }
                res_all[m_total + a] = observed[a] - z_mean[a];
            }
        } else if kind == 2 {
            let mount_q = normalize(mul(config.misalignment[i], config.orientation[i]));
            let mut z_pts = [[0.0; 4]; SIGMA_COUNT];
            for k in 0..SIGMA_COUNT {
                z_pts[k] = normalize(mul(mount_q, q_pts[k]));
            }
            let z_mean = quaternion_mean(&z_pts, &w_m);
            let z_mean_conj = conjugate(z_mean);
            for k in 0..SIGMA_COUNT {
                let dz = quaternion_to_rotation_vector(mul(z_pts[k], z_mean_conj));
                dz_t[m_total][k] = dz[0];
                dz_t[m_total + 1][k] = dz[1];
                dz_t[m_total + 2][k] = dz[2];
            }
            let obs_q = normalize([observed[0], observed[1], observed[2], observed[3]]);
            let res = quaternion_to_rotation_vector(mul(obs_q, z_mean_conj));
            res_all[m_total] = res[0];
            res_all[m_total + 1] = res[1];
            res_all[m_total + 2] = res[2];
        } else {
            let mut z_pts = [[0.0; 3]; SIGMA_COUNT];
            let mut all_finite = true;
            for k in 0..SIGMA_COUNT {
                let (_, z_k) = direction_prediction(config, i, q_pts[k])?;
                if z_k[..meas_width].iter().any(|x| !x.is_finite()) {
                    all_finite = false;
                    break;
                }
                z_pts[k] = z_k;
            }
            if !all_finite {
                continue;
            }
            if kind == 4 || kind == 5 {
                for a in 0..meas_width {
                    let mut delta_sum = 0.0;
                    for k in 0..SIGMA_COUNT {
                        delta_sum += w_m[k] * wrap_angle(z_pts[k][a] - z_pts[0][a]);
                    }
                    let z_mean_a = wrap_angle(z_pts[0][a] + delta_sum);
                    for k in 0..SIGMA_COUNT {
                        dz_t[m_total + a][k] = wrap_angle(z_pts[k][a] - z_mean_a);
                    }
                    res_all[m_total + a] = wrap_angle(observed[a] - z_mean_a);
                }
            } else {
                for a in 0..meas_width {
                    let mut sum = 0.0;
                    for k in 0..SIGMA_COUNT {
                        sum += w_m[k] * z_pts[k][a];
                    }
                    let z_mean_a = sum;
                    for k in 0..SIGMA_COUNT {
                        dz_t[m_total + a][k] = z_pts[k][a] - z_mean_a;
                    }
                    res_all[m_total + a] = observed[a] - z_mean_a;
                }
            }
        }

        let scalars_i = config.scalars.get(i).copied().unwrap_or([0.0; 3]);
        let r = sensor_noise_matrix(kind, meas_width, &config.noise_cholesky[i], scalars_i);
        for a in 0..meas_width {
            for b in 0..meas_width {
                p_zz[m_total + a][m_total + b] = r[a][b];
            }
        }
        m_total += meas_width;
    }

    if m_total > 0 {
        for a in 0..m_total {
            for b in a..m_total {
                let mut sum = p_zz[a][b];
                for k in 0..SIGMA_COUNT {
                    sum += w_c[k] * dz_t[a][k] * dz_t[b][k];
                }
                let avg = if a == b {
                    sum
                } else {
                    let mut sum_ba = p_zz[b][a];
                    for k in 0..SIGMA_COUNT {
                        sum_ba += w_c[k] * dz_t[b][k] * dz_t[a][k];
                    }
                    0.5 * (sum + sum_ba)
                };
                p_zz[a][b] = avg;
                p_zz[b][a] = avg;
            }
        }

        let mut p_xz = [[0.0; MAX_BATCH_MEAS]; 6];
        for a in 0..6 {
            for b in 0..m_total {
                let mut sum = 0.0;
                for k in 0..SIGMA_COUNT {
                    sum += w_c[k] * dx_pts[k][a] * dz_t[b][k];
                }
                p_xz[a][b] = sum;
            }
        }

        let sinv =
            invert_submatrix(&p_zz, m_total).ok_or("innovation covariance is singular")?;
        let mut gain = [[0.0; MAX_BATCH_MEAS]; 6];
        for a in 0..6 {
            for b in 0..m_total {
                let mut sum = 0.0;
                for k in 0..m_total {
                    sum += p_xz[a][k] * sinv[k][b];
                }
                gain[a][b] = sum;
            }
        }

        let mut delta = [0.0; 6];
        for a in 0..6 {
            let mut sum = 0.0;
            for b in 0..m_total {
                sum += gain[a][b] * res_all[b];
            }
            delta[a] = sum;
        }

        let p_prior = p;
        q = normalize(mul(rotation_to_q([delta[0], delta[1], delta[2]]), q));
        w = [w[0] + delta[3], w[1] + delta[4], w[2] + delta[5]];

        let mut gp = [[0.0; MAX_BATCH_MEAS]; 6];
        for a in 0..6 {
            for b in 0..m_total {
                let mut sum = 0.0;
                for k in 0..m_total {
                    sum += gain[a][k] * p_zz[k][b];
                }
                gp[a][b] = sum;
            }
        }
        let mut p_raw = [[0.0; 6]; 6];
        for a in 0..6 {
            for b in 0..6 {
                let mut sum = 0.0;
                for k in 0..m_total {
                    sum += gp[a][k] * gain[b][k];
                }
                p_raw[a][b] = p[a][b] - sum;
            }
        }

        let mut reset = identity6();
        let sk = skew([delta[0], delta[1], delta[2]]);
        for i in 0..3 {
            for j in 0..3 {
                reset[i][j] += 0.5 * sk[i][j];
            }
        }
        p = sym6(matmul6(matmul6(reset, p_raw), transpose6(reset)));

        if let Some(out) = diagnostics.as_deref_mut() {
            let diag_w = m_total.min(3);
            for a in 0..3 {
                out[43 + a] = if a < diag_w { res_all[a] } else { f64::NAN };
                for b in 0..3 {
                    out[46 + a * 3 + b] = if a < diag_w && b < diag_w {
                        p_zz[a][b]
                    } else {
                        f64::NAN
                    };
                }
            }

            let mut nis = 0.0;
            for a in 0..m_total {
                for b in 0..m_total {
                    nis += res_all[a] * sinv[a][b] * res_all[b];
                }
            }
            out[55] = nis;

            let p_prior_inv =
                invert_submatrix(&p_prior, 6).ok_or("prior covariance is singular")?;
            let mut p_inv_delta = [0.0; 6];
            for k in 0..6 {
                let mut sum = 0.0;
                for c in 0..6 {
                    sum += p_prior_inv[c][k] * delta[c];
                }
                p_inv_delta[k] = sum;
            }
            for a in 0..3 {
                if a < diag_w {
                    let mut h_delta = 0.0;
                    for k in 0..6 {
                        h_delta += p_xz[k][a] * p_inv_delta[k];
                    }
                    out[57 + a] = res_all[a] - h_delta;
                } else {
                    out[57 + a] = f64::NAN;
                }
            }

            for a in 0..6 {
                for b in 0..3 {
                    out[60 + a * 3 + b] = if b < diag_w { gain[a][b] } else { f64::NAN };
                }
            }
        }
    }

    Ok((q, w, p))
}

/// Predict and batch-consume all valid attitude sensor readings from prepared sensor rows.
#[allow(clippy::too_many_arguments)]
pub fn step(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    acceleration_psd: Mat3,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    step_impl(
        q,
        w,
        p,
        acceleration_psd,
        dt,
        readings,
        config,
        alpha,
        beta,
        kappa,
        None,
    )
}

/// One subsystem tick, including the diagnostics of the batched correction.
/// Diagnostic layout matches the shared 96-element 6-state simulation result fields.
pub fn step_diagnostics(
    state: &mut AttitudeBatchedUKF,
    dt: f64,
    readings: &[f64],
    config: &SensorConfig,
    truth: &[f64],
    nees: bool,
) -> Result<[f64; 96], &'static str> {
    let mut out = [f64::NAN; 96];
    let (q, w, p) = step_impl(
        state.base.quaternion,
        state.base.rate,
        state.base.covariance,
        state.base.acceleration_psd,
        dt,
        readings,
        config,
        state.alpha,
        state.beta,
        state.kappa,
        Some(&mut out),
    )?;

    out[..4].copy_from_slice(&q);
    out[4..7].copy_from_slice(&w);
    for a in 0..6 {
        for b in 0..6 {
            out[7 + a * 6 + b] = p[a][b];
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            out[78 + a * 3 + b] = p[a][b];
            out[87 + a * 3 + b] = p[a + 3][b + 3];
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
        let l = cholesky6(p).map_err(|_| "estimator covariance is not positive definite")?;
        let mut z = [0.0; 6];
        for a in 0..6 {
            let mut sum = error[a];
            for b in 0..a {
                sum -= l[a][b] * z[b];
            }
            z[a] = sum / l[a][a];
        }
        out[56] = z.iter().map(|x| x * x).sum();
    }
    state.base.quaternion = q;
    state.base.rate = w;
    state.base.covariance = p;
    Ok(out)
}

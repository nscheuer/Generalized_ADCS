//! Six-error-state unscented Kalman filter (UKF) with stack-allocated arrays.

use std::f64::consts::PI;

use super::attitude_mekf::SensorConfig;
use super::base::{AttitudeEstimator, Mat3, Mat6};
use super::direction_correction::{direction_prediction, inverse_submatrix};

pub const COVARIANCE_SIZE: usize = 6;
const SIGMA_COUNT: usize = 2 * COVARIANCE_SIZE + 1;

#[inline]
fn chart_limit_angle() -> f64 {
    2.0 * 0.95_f64.asin()
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeSequentialUKF {
    pub base: AttitudeEstimator,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl AttitudeSequentialUKF {
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

fn validate_ukf_params(alpha: f64, beta: f64, kappa: f64) -> Result<f64, &'static str> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err("alpha must be finite and positive");
    }
    if !beta.is_finite() {
        return Err("beta must be finite");
    }
    if !kappa.is_finite() {
        return Err("kappa must be finite");
    }
    let scale = alpha * alpha * (COVARIANCE_SIZE as f64 + kappa);
    if scale <= 0.0 {
        return Err("alpha and kappa must give a positive UKF scale");
    }
    Ok(scale)
}

fn ukf_weights(alpha: f64, beta: f64, kappa: f64) -> Result<(f64, [f64; SIGMA_COUNT], [f64; SIGMA_COUNT]), &'static str> {
    let scale = validate_ukf_params(alpha, beta, kappa)?;
    let n = COVARIANCE_SIZE as f64;
    let lam = scale - n;
    let gamma = scale.sqrt();
    let mut w_m = [0.5 / scale; SIGMA_COUNT];
    let mut w_c = [0.5 / scale; SIGMA_COUNT];
    w_m[0] = lam / scale;
    w_c[0] = lam / scale + (1.0 - alpha * alpha + beta);
    Ok((gamma, w_m, w_c))
}

fn jacobi_eigh6(mut a: Mat6) -> ([f64; 6], Mat6) {
    let mut v = identity6();
    for _ in 0..(64 * 6 * 6) {
        let mut p = 0;
        let mut q = 1;
        let mut largest = 0.0_f64;
        for i in 0..6 {
            for j in i + 1..6 {
                let val = a[i][j].abs();
                if val > largest {
                    largest = val;
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1e-15 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..6 {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        for k in 0..6 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let vals = std::array::from_fn(|i| a[i][i]);
    (vals, v)
}

fn cholesky6(a: Mat6) -> Result<Mat6, &'static str> {
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err("covariance matrix is not positive definite");
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

fn cholesky_psd(p: Mat6) -> Result<Mat6, &'static str> {
    let p_sym = sym6(p);
    let (mut vals, vecs) = jacobi_eigh6(p_sym);
    let min_val = vals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_val > 1.0e-15 {
        return cholesky6(p_sym);
    }
    for v in &mut vals {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut reconstructed = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = if i == j { 1.0e-14 } else { 0.0 };
            for k in 0..6 {
                sum += vecs[i][k] * vals[k] * vecs[j][k];
            }
            reconstructed[i][j] = sum;
        }
    }
    cholesky6(sym6(reconstructed))
}

/// Weighted manifold mean solving sum(w_i * log(q_i * mean^-1)) = 0.
pub fn quaternion_mean(quaternions: &[[f64; 4]; SIGMA_COUNT], weights: &[f64; SIGMA_COUNT]) -> [f64; 4] {
    let mut normalized = [[0.0; 4]; SIGMA_COUNT];
    for i in 0..SIGMA_COUNT {
        normalized[i] = normalize(quaternions[i]);
    }
    let mut mean = normalized[0];
    let mut deviations = [[0.0; 3]; SIGMA_COUNT];
    for _ in 0..32 {
        let mean_conj = conjugate(mean);
        let mut scale = 0.0;
        let mut residual = [0.0; 3];
        for i in 0..SIGMA_COUNT {
            let dev = quaternion_to_rotation_vector(mul(normalized[i], mean_conj));
            deviations[i] = dev;
            let dev_norm = dot3(dev, dev).sqrt();
            scale += weights[i].abs() * dev_norm;
            residual[0] += weights[i] * dev[0];
            residual[1] += weights[i] * dev[1];
            residual[2] += weights[i] * dev[2];
        }
        let residual_norm = dot3(residual, residual).sqrt();
        let tolerance = 1.0e-12 + 64.0 * f64::EPSILON * scale;
        if residual_norm <= tolerance {
            return mean;
        }
        let factor = 1.0_f64.min((0.5 * PI) / residual_norm.max(1.0e-15));
        let step = [
            residual[0] * factor,
            residual[1] * factor,
            residual[2] * factor,
        ];
        let mut improved = false;
        let mut alpha = 1.0;
        for _ in 0..16 {
            let candidate = normalize(mul(
                rotation_to_q([alpha * step[0], alpha * step[1], alpha * step[2]]),
                mean,
            ));
            let cand_conj = conjugate(candidate);
            let mut cand_residual = [0.0; 3];
            for i in 0..SIGMA_COUNT {
                let dev = quaternion_to_rotation_vector(mul(normalized[i], cand_conj));
                cand_residual[0] += weights[i] * dev[0];
                cand_residual[1] += weights[i] * dev[1];
                cand_residual[2] += weights[i] * dev[2];
            }
            if dot3(cand_residual, cand_residual).sqrt() < residual_norm {
                mean = candidate;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            break;
        }
    }
    mean
}

#[allow(clippy::type_complexity)]
fn sigma_points(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> Result<
    (
        [[f64; 4]; SIGMA_COUNT],
        [[f64; 3]; SIGMA_COUNT],
        [f64; SIGMA_COUNT],
        [f64; SIGMA_COUNT],
    ),
    &'static str,
> {
    let (gamma, w_m, w_c) = ukf_weights(alpha, beta, kappa)?;
    let chol = cholesky_psd(p)?;
    let mut q_pts = [[0.0; 4]; SIGMA_COUNT];
    let mut w_pts = [[0.0; 3]; SIGMA_COUNT];
    q_pts[0] = normalize(q);
    w_pts[0] = w;
    let limit = chart_limit_angle();
    for i in 0..COVARIANCE_SIZE {
        let mut d_theta = [gamma * chol[0][i], gamma * chol[1][i], gamma * chol[2][i]];
        let d_omega = [gamma * chol[3][i], gamma * chol[4][i], gamma * chol[5][i]];
        let theta_norm = dot3(d_theta, d_theta).sqrt();
        if theta_norm > limit {
            let s = limit / theta_norm;
            d_theta = [d_theta[0] * s, d_theta[1] * s, d_theta[2] * s];
        }
        let dq = rotation_to_q(d_theta);
        q_pts[1 + i] = normalize(mul(dq, q_pts[0]));
        w_pts[1 + i] = [w[0] + d_omega[0], w[1] + d_omega[1], w[2] + d_omega[2]];
        q_pts[1 + COVARIANCE_SIZE + i] = normalize(mul(conjugate(dq), q_pts[0]));
        w_pts[1 + COVARIANCE_SIZE + i] = [w[0] - d_omega[0], w[1] - d_omega[1], w[2] - d_omega[2]];
    }
    Ok((q_pts, w_pts, w_m, w_c))
}

fn state_deviations(
    q_pts: &[[f64; 4]; SIGMA_COUNT],
    w_pts: &[[f64; 3]; SIGMA_COUNT],
    q_ref: [f64; 4],
    w_ref: [f64; 3],
) -> [[f64; 6]; SIGMA_COUNT] {
    let mut deviations = [[0.0; 6]; SIGMA_COUNT];
    let q_ref_conj = conjugate(q_ref);
    for k in 0..SIGMA_COUNT {
        let d_theta = quaternion_to_rotation_vector(mul(q_pts[k], q_ref_conj));
        deviations[k][0] = d_theta[0];
        deviations[k][1] = d_theta[1];
        deviations[k][2] = d_theta[2];
        deviations[k][3] = w_pts[k][0] - w_ref[0];
        deviations[k][4] = w_pts[k][1] - w_ref[1];
        deviations[k][5] = w_pts[k][2] - w_ref[2];
    }
    deviations
}

/// Propagate sigma points and add continuous white angular-acceleration noise.
pub fn predict(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    acceleration_psd: Mat3,
    dt: f64,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> Result<([f64; 4], [f64; 3], Mat6), &'static str> {
    if dt < 0.0 {
        return Err("dt must be nonnegative");
    }
    let (q_pts, w_pts, w_m, w_c) = sigma_points(q, w, p, alpha, beta, kappa)?;
    let mut q_prop = [[0.0; 4]; SIGMA_COUNT];
    let mut w_mean = [0.0; 3];
    for k in 0..SIGMA_COUNT {
        let increment = rotation_to_q([-w_pts[k][0] * dt, -w_pts[k][1] * dt, -w_pts[k][2] * dt]);
        q_prop[k] = normalize(mul(increment, q_pts[k]));
        w_mean[0] += w_m[k] * w_pts[k][0];
        w_mean[1] += w_m[k] * w_pts[k][1];
        w_mean[2] += w_m[k] * w_pts[k][2];
    }
    let q_mean = quaternion_mean(&q_prop, &w_m);
    let deviations = state_deviations(&q_prop, &w_pts, q_mean, w_mean);

    let dt2 = dt * dt * 0.5;
    let dt3 = dt * dt * dt / 3.0;
    let mut p_pred = [[0.0; 6]; 6];
    for a in 0..6 {
        for b in 0..6 {
            let mut sum = 0.0;
            for k in 0..SIGMA_COUNT {
                sum += w_c[k] * deviations[k][a] * deviations[k][b];
            }
            p_pred[a][b] = sum;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            p_pred[i][j] += acceleration_psd[i][j] * dt3;
            let cross = -acceleration_psd[i][j] * dt2;
            p_pred[i][j + 3] += cross;
            p_pred[i + 3][j] += cross;
            p_pred[i + 3][j + 3] += acceleration_psd[i][j] * dt;
        }
    }
    Ok((q_mean, w_mean, sym6(p_pred)))
}

struct UnscentedCorrection {
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    gain: [[f64; 3]; 6],
    p_zz: Mat3,
}

fn correct_unscented(
    q: [f64; 4],
    w: [f64; 3],
    p: Mat6,
    dx_pts: &[[f64; 6]; SIGMA_COUNT],
    dz_pts: &[[f64; 3]; SIGMA_COUNT],
    w_c: &[f64; SIGMA_COUNT],
    r: Mat3,
    residual: [f64; 3],
    width: usize,
) -> Result<UnscentedCorrection, &'static str> {
    let mut p_zz = [[0.0; 3]; 3];
    for a in 0..width {
        for b in 0..width {
            let mut sum = r[a][b];
            for k in 0..SIGMA_COUNT {
                sum += w_c[k] * dz_pts[k][a] * dz_pts[k][b];
            }
            p_zz[a][b] = sum;
        }
    }
    for a in 0..width {
        for b in a + 1..width {
            let avg = 0.5 * (p_zz[a][b] + p_zz[b][a]);
            p_zz[a][b] = avg;
            p_zz[b][a] = avg;
        }
    }

    let mut p_xz = [[0.0; 3]; 6];
    for a in 0..6 {
        for b in 0..width {
            let mut sum = 0.0;
            for k in 0..SIGMA_COUNT {
                sum += w_c[k] * dx_pts[k][a] * dz_pts[k][b];
            }
            p_xz[a][b] = sum;
        }
    }

    let sinv = inverse_submatrix(p_zz, width).ok_or("innovation covariance is singular")?;
    let mut gain = [[0.0; 3]; 6];
    for a in 0..6 {
        for b in 0..width {
            let mut sum = 0.0;
            for k in 0..width {
                sum += p_xz[a][k] * sinv[k][b];
            }
            gain[a][b] = sum;
        }
    }

    let mut delta = [0.0; 6];
    for a in 0..6 {
        let mut sum = 0.0;
        for b in 0..width {
            sum += gain[a][b] * residual[b];
        }
        delta[a] = sum;
    }

    let q_next = normalize(mul(rotation_to_q([delta[0], delta[1], delta[2]]), q));
    let w_next = [w[0] + delta[3], w[1] + delta[4], w[2] + delta[5]];

    // p_raw = p - gain * p_zz * gain^T
    let mut gp = [[0.0; 3]; 6];
    for a in 0..6 {
        for b in 0..width {
            let mut sum = 0.0;
            for k in 0..width {
                sum += gain[a][k] * p_zz[k][b];
            }
            gp[a][b] = sum;
        }
    }
    let mut p_raw = [[0.0; 6]; 6];
    for a in 0..6 {
        for b in 0..6 {
            let mut sum = 0.0;
            for k in 0..width {
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
    let p_next = sym6(matmul6(matmul6(reset, p_raw), transpose6(reset)));

    Ok(UnscentedCorrection {
        q: q_next,
        w: w_next,
        p: p_next,
        gain,
        p_zz,
    })
}

fn sensor_noise_matrix(kind: i64, width: usize, chol_i: &[[f64; 6]; 6], scalars_i: [f64; 3]) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    if kind == 3 {
        r[0][0] = scalars_i[2] * scalars_i[2];
        return r;
    }
    for a in 0..width {
        for b in 0..width {
            let mut sum = 0.0;
            for k in 0..width {
                sum += chol_i[a][k] * chol_i[b][k];
            }
            r[a][b] = sum;
        }
    }
    r
}

#[inline]
fn wrap_angle(v: f64) -> f64 {
    (v + PI).rem_euclid(2.0 * PI) - PI
}

fn record_measurement_diagnostics(
    out: &mut [f64; 96],
    width: usize,
    residual: [f64; 3],
    p_zz: Mat3,
    post_residual: [f64; 3],
    gain: [[f64; 3]; 6],
) -> Result<(), &'static str> {
    for a in 0..3 {
        out[43 + a] = if a < width { residual[a] } else { f64::NAN };
        out[57 + a] = if a < width { post_residual[a] } else { f64::NAN };
        for b in 0..3 {
            out[46 + a * 3 + b] = if a < width && b < width {
                p_zz[a][b]
            } else {
                f64::NAN
            };
        }
    }
    let sinv = inverse_submatrix(p_zz, width).ok_or("innovation covariance is singular")?;
    let mut nis = 0.0;
    for a in 0..width {
        for b in 0..width {
            nis += residual[a] * sinv[a][b] * residual[b];
        }
    }
    out[55] = nis;
    for a in 0..6 {
        for b in 0..3 {
            out[60 + a * 3 + b] = if b < width { gain[a][b] } else { f64::NAN };
        }
    }
    Ok(())
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
    let (mut q, mut w, mut p) = predict(q_init, w_init, p_init, acceleration_psd, dt, alpha, beta, kappa)?;

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

        let (q_pts, w_pts, w_m, w_c) = sigma_points(q, w, p, alpha, beta, kappa)?;
        let dx_pts = state_deviations(&q_pts, &w_pts, q, w);
        let meas_width = if matches!(kind, 0 | 1 | 2) { 3 } else { obs_width };
        let scalars_i = config.scalars.get(i).copied().unwrap_or([0.0; 3]);
        let r = sensor_noise_matrix(kind, meas_width, &config.noise_cholesky[i], scalars_i);

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
            let mut dz_pts = [[0.0; 3]; SIGMA_COUNT];
            for k in 0..SIGMA_COUNT {
                dz_pts[k] = [
                    z_pts[k][0] - z_mean[0],
                    z_pts[k][1] - z_mean[1],
                    z_pts[k][2] - z_mean[2],
                ];
            }
            let residual = [
                observed[0] - z_mean[0],
                observed[1] - z_mean[1],
                observed[2] - z_mean[2],
            ];
            let corr = correct_unscented(q, w, p, &dx_pts, &dz_pts, &w_c, r, residual, 3)?;
            if let Some(out) = diagnostics.as_deref_mut() {
                let post_rot = mul3v(mount, corr.w);
                let post_res = [
                    observed[0] - (post_rot[0] + config.bias[i][0]),
                    observed[1] - (post_rot[1] + config.bias[i][1]),
                    observed[2] - (post_rot[2] + config.bias[i][2]),
                ];
                record_measurement_diagnostics(out, 3, residual, corr.p_zz, post_res, corr.gain)?;
            }
            q = corr.q;
            w = corr.w;
            p = corr.p;
        } else if kind == 2 {
            let mount_q = normalize(mul(config.misalignment[i], config.orientation[i]));
            let mut z_pts = [[0.0; 4]; SIGMA_COUNT];
            for k in 0..SIGMA_COUNT {
                z_pts[k] = normalize(mul(mount_q, q_pts[k]));
            }
            let z_mean = quaternion_mean(&z_pts, &w_m);
            let z_mean_conj = conjugate(z_mean);
            let mut dz_pts = [[0.0; 3]; SIGMA_COUNT];
            for k in 0..SIGMA_COUNT {
                dz_pts[k] = quaternion_to_rotation_vector(mul(z_pts[k], z_mean_conj));
            }
            let obs_q = normalize([observed[0], observed[1], observed[2], observed[3]]);
            let residual = quaternion_to_rotation_vector(mul(obs_q, z_mean_conj));
            let corr = correct_unscented(q, w, p, &dx_pts, &dz_pts, &w_c, r, residual, 3)?;
            if let Some(out) = diagnostics.as_deref_mut() {
                let post_pred = normalize(mul(mount_q, corr.q));
                let post_res = quaternion_to_rotation_vector(mul(obs_q, conjugate(post_pred)));
                record_measurement_diagnostics(out, 3, residual, corr.p_zz, post_res, corr.gain)?;
            }
            q = corr.q;
            w = corr.w;
            p = corr.p;
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

            let mut dz_pts = [[0.0; 3]; SIGMA_COUNT];
            let mut residual = [0.0; 3];
            if kind == 4 || kind == 5 {
                let mut z_mean = [0.0; 3];
                for a in 0..meas_width {
                    let mut delta_sum = 0.0;
                    for k in 0..SIGMA_COUNT {
                        delta_sum += w_m[k] * wrap_angle(z_pts[k][a] - z_pts[0][a]);
                    }
                    z_mean[a] = wrap_angle(z_pts[0][a] + delta_sum);
                    for k in 0..SIGMA_COUNT {
                        dz_pts[k][a] = wrap_angle(z_pts[k][a] - z_mean[a]);
                    }
                    residual[a] = wrap_angle(observed[a] - z_mean[a]);
                }
            } else {
                let mut z_mean = [0.0; 3];
                for a in 0..meas_width {
                    let mut sum = 0.0;
                    for k in 0..SIGMA_COUNT {
                        sum += w_m[k] * z_pts[k][a];
                    }
                    z_mean[a] = sum;
                    for k in 0..SIGMA_COUNT {
                        dz_pts[k][a] = z_pts[k][a] - z_mean[a];
                    }
                    residual[a] = observed[a] - z_mean[a];
                }
            }

            let corr = correct_unscented(q, w, p, &dx_pts, &dz_pts, &w_c, r, residual, meas_width)?;
            if let Some(out) = diagnostics.as_deref_mut() {
                let (_, post_pred) = direction_prediction(config, i, corr.q)?;
                let mut post_res = [f64::NAN; 3];
                if post_pred[..meas_width].iter().all(|x| x.is_finite()) {
                    for a in 0..meas_width {
                        let diff = observed[a] - post_pred[a];
                        post_res[a] = if kind == 4 || kind == 5 {
                            wrap_angle(diff)
                        } else {
                            diff
                        };
                    }
                }
                record_measurement_diagnostics(
                    out,
                    meas_width,
                    residual,
                    corr.p_zz,
                    post_res,
                    corr.gain,
                )?;
            }
            q = corr.q;
            w = corr.w;
            p = corr.p;
        }
    }

    Ok((q, w, p))
}

/// Predict and process valid attitude sensor readings from prepared sensor rows.
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

/// One subsystem tick, including the diagnostics of the last valid correction.
/// Diagnostic layout matches the shared 96-element 6-state simulation result fields.
pub fn step_diagnostics(
    state: &mut AttitudeSequentialUKF,
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

fn skew(v: [f64; 3]) -> Mat3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn mul3v(a: Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn identity6() -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
}

fn matmul6(a: Mat6, b: Mat6) -> Mat6 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i][k] * b[k][j];
            }
            s
        })
    })
}

fn transpose6(a: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn sym6(a: Mat6) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (a[i][j] + a[j][i])))
}

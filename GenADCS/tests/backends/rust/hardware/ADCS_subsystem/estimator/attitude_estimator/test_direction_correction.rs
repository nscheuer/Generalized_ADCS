use super::attitude_ekf::{self, Mat7};
use super::attitude_mekf::{self, SensorConfig};
use super::base::Mat6;
use super::direction_correction::{correct_direction, direction_prediction};

fn eye6(scale: f64) -> Mat6 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
}

fn eye7(scale: f64) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
}

fn chol_diag(vals: [f64; 3]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..3 {
        c[i][i] = vals[i];
    }
    c
}

#[test]
fn magnetometer_prediction_and_correction_work_for_mekf_and_ekf() {
    let q_true = [1.0, 0.0, 0.0, 0.0];
    let q_err = [0.995, 0.07, -0.05, 0.04];
    let mag_ref = [0.3, -0.4, 0.86];
    let config = SensorConfig {
        code: vec![1],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]],
        bias: vec![[0.01, -0.02, 0.005, 0.0, 0.0, 0.0]],
        noise_cholesky: vec![chol_diag([1e-3, 1e-3, 1e-3])],
        normal: vec![[0.0, 0.0, 1.0]],
        fov: vec![[0.0, 0.0]],
        scalars: vec![[10.0, 0.0, 0.0]],
        enabled: vec![true],
        offsets: vec![0],
        reference: vec![mag_ref],
    };

    let (w_pred, obs) = direction_prediction(&config, 0, q_true).unwrap();
    assert_eq!(w_pred, 3);
    assert!((obs[0] - (mag_ref[0] + 0.01)).abs() < 1e-12);

    let mekf_corr = correct_direction::<6>(q_err, [0.0; 3], eye6(0.1), &config, 0, &obs[..3])
        .unwrap()
        .unwrap();
    let (_, pred_after_mekf) = direction_prediction(&config, 0, mekf_corr.quaternion).unwrap();
    let err_before: f64 = {
        let (_, pb) = direction_prediction(&config, 0, q_err).unwrap();
        (0..3).map(|i| (pb[i] - obs[i]).powi(2)).sum::<f64>().sqrt()
    };
    let err_after_mekf: f64 = (0..3)
        .map(|i| (pred_after_mekf[i] - obs[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err_after_mekf < err_before * 0.1);

    let ekf_corr = correct_direction::<7>(q_err, [0.0; 3], eye7(0.1), &config, 0, &obs[..3])
        .unwrap()
        .unwrap();
    let (_, pred_after_ekf) = direction_prediction(&config, 0, ekf_corr.quaternion).unwrap();
    let err_after_ekf: f64 = (0..3)
        .map(|i| (pred_after_ekf[i] - obs[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err_after_ekf < err_before * 0.1);
}

#[test]
fn coarse_sun_sensor_prediction_and_correction_use_noise_std() {
    let sun_ref = [0.2, 0.1, 0.9746794344808963];
    let q_err = [0.996, 0.06, -0.04, 0.02];
    let config = SensorConfig {
        code: vec![3],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]],
        bias: vec![[0.0; 6]],
        noise_cholesky: vec![[[0.0; 6]; 6]],
        normal: vec![[0.0, 0.0, 1.0]],
        fov: vec![[0.0, 0.0]],
        scalars: vec![[5.0, 2.0, 0.01]],
        enabled: vec![true],
        offsets: vec![0],
        reference: vec![sun_ref],
    };

    let (w_pred, obs) = direction_prediction(&config, 0, [1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(w_pred, 1);
    assert!((obs[0] - 2.0 * sun_ref[2]).abs() < 1e-12);

    let corr6 = correct_direction::<6>(q_err, [0.0; 3], eye6(0.1), &config, 0, &obs[..1])
        .unwrap()
        .unwrap();
    assert!((corr6.innovation_covariance[0][0] - (0.0001)).abs() > 0.0);
    let (_, pred6) = direction_prediction(&config, 0, corr6.quaternion).unwrap();
    let (_, pb) = direction_prediction(&config, 0, q_err).unwrap();
    assert!((pred6[0] - obs[0]).abs() < (pb[0] - obs[0]).abs());

    let corr7 = correct_direction::<7>(q_err, [0.0; 3], eye7(0.1), &config, 0, &obs[..1])
        .unwrap()
        .unwrap();
    let (_, pred7) = direction_prediction(&config, 0, corr7.quaternion).unwrap();
    assert!((pred7[0] - obs[0]).abs() < (pb[0] - obs[0]).abs());
}

#[test]
fn fine_sun_and_earth_sensors_gate_fov_and_update_attitude() {
    let dir_ref = [0.05, -0.04, 0.9979478944313676];
    let q_err = [0.996, 0.05, 0.05, 0.01];
    let config = SensorConfig {
        code: vec![4, 5],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        bias: vec![[0.0; 6]; 2],
        noise_cholesky: vec![chol_diag([1e-3, 1e-3, 1e-3]); 2],
        normal: vec![[0.0, 0.0, 1.0]; 2],
        fov: vec![[0.5, 0.5], [0.01, 0.01]], // sensor 0 in FOV, sensor 1 out of FOV
        scalars: vec![[0.0; 3]; 2],
        enabled: vec![true, true],
        offsets: vec![0, 2],
        reference: vec![dir_ref, dir_ref],
    };

    let (w0, obs0) = direction_prediction(&config, 0, [1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(w0, 2);
    assert!(obs0[0].is_finite() && obs0[1].is_finite());

    let (_, obs1_oob) = direction_prediction(&config, 1, [1.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(obs1_oob[0].is_nan() && obs1_oob[1].is_nan());
    assert!(correct_direction::<6>(q_err, [0.0; 3], eye6(0.1), &config, 1, &[0.0, 0.0])
        .unwrap()
        .is_none());

    // Step MEKF & EKF with both sensors (code 4 updates, code 5 is skipped due to FOV)
    let readings = [obs0[0], obs0[1], 0.0, 0.0];
    let psd = [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]];
    let (q_m, _, _) =
        attitude_mekf::step(q_err, [0.0; 3], eye6(0.1), psd, 0.0, &readings, &config).unwrap();
    let (q_e, _, _) =
        attitude_ekf::step(q_err, [0.0; 3], eye7(0.1), psd, 0.0, &readings, &config).unwrap();

    let (_, pb) = direction_prediction(&config, 0, q_err).unwrap();
    let (_, pm) = direction_prediction(&config, 0, q_m).unwrap();
    let (_, pe) = direction_prediction(&config, 0, q_e).unwrap();
    let err_before = ((pb[0] - obs0[0]).powi(2) + (pb[1] - obs0[1]).powi(2)).sqrt();
    let err_m = ((pm[0] - obs0[0]).powi(2) + (pm[1] - obs0[1]).powi(2)).sqrt();
    let err_e = ((pe[0] - obs0[0]).powi(2) + (pe[1] - obs0[1]).powi(2)).sqrt();
    assert!(err_m < err_before * 0.1);
    assert!(err_e < err_before * 0.1);
}

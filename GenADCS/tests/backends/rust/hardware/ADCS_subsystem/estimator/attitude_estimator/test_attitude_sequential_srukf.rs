use super::attitude_mekf::SensorConfig;
use super::attitude_sequential_srukf::{
    predict, quaternion_mean, step, step_diagnostics, AttitudeSequentialSRUKF, COVARIANCE_SIZE,
};
use super::attitude_sequential_ukf::{
    predict as ukf_predict, step as ukf_step, step_diagnostics as ukf_step_diagnostics,
    AttitudeSequentialUKF,
};
use super::base::AttitudeEstimator;

fn eye6(scale: f64) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
}

fn make_base() -> AttitudeEstimator {
    AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [0.1, -0.05, 0.02],
        eye6(0.1),
        [[1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]],
        vec![],
    )
    .unwrap()
}

fn make_all_sensors_config() -> SensorConfig {
    SensorConfig {
        code: vec![0, 1, 2, 3, 4, 5],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        bias: vec![[0.0; 6]; 6],
        noise_cholesky: vec![eye6(1e-3); 6],
        normal: vec![[0.0, 0.0, 1.0]; 6],
        fov: vec![
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        scalars: vec![
            [0.0; 3],
            [10.0, 0.0, 0.0],
            [0.0; 3],
            [10.0, 1.0, 1e-3],
            [0.0; 3],
            [0.0; 3],
        ],
        enabled: vec![true; 6],
        offsets: vec![0, 3, 6, 10, 11, 13],
        reference: vec![
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0; 3],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    }
}

#[test]
fn srukf_validates_hyperparameters_and_exposes_params() {
    let base = make_base();
    assert_eq!(AttitudeSequentialSRUKF::COVARIANCE_SIZE, 6);
    assert_eq!(COVARIANCE_SIZE, 6);

    let srukf = AttitudeSequentialSRUKF::new(base.clone(), 0.5, 2.0, 1.0).unwrap();
    assert_eq!(srukf.ukf_params(), [0.5, 2.0, 1.0]);

    assert!(AttitudeSequentialSRUKF::new(base.clone(), 0.0, 2.0, 0.0).is_err());
    assert!(AttitudeSequentialSRUKF::new(base.clone(), -0.5, 2.0, 0.0).is_err());
    assert!(AttitudeSequentialSRUKF::new(base.clone(), f64::NAN, 2.0, 0.0).is_err());
    assert!(AttitudeSequentialSRUKF::new(base.clone(), 1.0, f64::INFINITY, 0.0).is_err());
    assert!(AttitudeSequentialSRUKF::new(base.clone(), 1.0, 2.0, f64::NAN).is_err());
    assert!(AttitudeSequentialSRUKF::new(base, 1.0, 2.0, -6.0).is_err());
}

#[test]
fn srukf_predict_propagates_state_and_process_noise() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let w = [0.0; 3];
    let p = [[0.0; 6]; 6];
    let psd = [[2e-5, 0.0, 0.0], [0.0, 2e-5, 0.0], [0.0, 0.0, 2e-5]];
    let (q1, w1, p1) = predict(q, w, p, psd, 2.0, 1.0, 2.0, 0.0).unwrap();
    assert!((q1[0] - 1.0).abs() < 1e-14);
    assert!(w1.iter().all(|x| x.abs() < 1e-14));
    assert!((p1[0][0] - 2e-5 * 8.0 / 3.0).abs() < 1e-12);
    assert!((p1[3][3] - 2e-5 * 2.0).abs() < 1e-12);
    assert!((p1[0][3] + 2e-5 * 2.0).abs() < 1e-12);
    assert!(predict(q, w, p, psd, -0.1, 1.0, 2.0, 0.0).is_err());
}

#[test]
fn srukf_quaternion_mean_averages_symmetric_perturbations() {
    let angle: f64 = 0.2;
    let half: f64 = 0.5 * angle;
    let q_pos = [half.cos(), half.sin(), 0.0, 0.0];
    let q_neg = [half.cos(), -half.sin(), 0.0, 0.0];
    let mut quats = [[1.0, 0.0, 0.0, 0.0]; 13];
    quats[1] = q_pos;
    quats[2] = q_neg;
    let mut weights = [0.0; 13];
    weights[0] = 0.5;
    weights[1] = 0.25;
    weights[2] = 0.25;
    let mean = quaternion_mean(&quats, &weights);
    assert!((mean[0] - 1.0).abs() < 1e-12);
    assert!(mean[1].abs() < 1e-12);
    assert!(mean[2].abs() < 1e-12);
    assert!(mean[3].abs() < 1e-12);
}

#[test]
fn srukf_step_consumes_all_six_sensor_types_and_skips_nan_and_fov_dropout() {
    let config = make_all_sensors_config();
    let q_init = [0.05_f64.cos(), 0.05_f64.sin(), 0.0, 0.0];
    let w_init = [0.1, -0.1, 0.05];
    let readings = [
        0.0, 0.0, 0.0, // gyro
        1.0, 0.0, 0.0, // magnetometer
        1.0, 0.0, 0.0, 0.0, // star tracker
        1.0, // coarse sun
        0.0, 0.0, // fine sun
        0.0, 0.0, // fine earth
    ];
    let (q1, w1, p1) = step(
        q_init,
        w_init,
        eye6(0.1),
        [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]],
        0.1,
        &readings,
        &config,
        1.0,
        2.0,
        0.0,
    )
    .unwrap();
    let init_err = q_init[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let final_err = q1[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(final_err < init_err);
    assert!(w1.iter().map(|x| x * x).sum::<f64>().sqrt() < 0.01);
    for i in 0..6 {
        for j in 0..6 {
            assert!((p1[i][j] - p1[j][i]).abs() < 1e-14);
        }
    }

    // NaN readings should be skipped cleanly without corrupting state.
    let nan_readings = [f64::NAN; 15];
    let (q_nan, w_nan, _) = step(
        q1,
        w1,
        p1,
        [[0.0; 3]; 3],
        0.1,
        &nan_readings,
        &config,
        1.0,
        2.0,
        0.0,
    )
    .unwrap();
    assert!(q_nan.iter().all(|x| x.is_finite()));
    for j in 0..3 {
        assert!((w_nan[j] - w1[j]).abs() < 1e-14);
    }

    // FOV-boundary sigma-point dropout: fine sun with tight FOV and large covariance skips update.
    let tight_fov_config = SensorConfig {
        code: vec![4],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]],
        bias: vec![[0.0; 6]],
        noise_cholesky: vec![eye6(1e-3)],
        normal: vec![[0.0, 0.0, 1.0]],
        fov: vec![[0.05, 0.05]],
        scalars: vec![[0.0; 3]],
        enabled: vec![true],
        offsets: vec![0],
        reference: vec![[0.0, 0.0, 1.0]],
    };
    let (q_fov, w_fov, _) = step(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        eye6(0.5),
        [[0.0; 3]; 3],
        0.0,
        &[0.02, 0.02],
        &tight_fov_config,
        1.0,
        2.0,
        0.0,
    )
    .unwrap();
    assert_eq!(q_fov, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(w_fov, [0.0; 3]);
}

#[test]
fn srukf_matches_ukf_parity_for_alpha_one_and_half() {
    let config = make_all_sensors_config();
    let q_init = [0.05_f64.cos(), 0.05_f64.sin(), -0.02, 0.01];
    let w_init = [0.1, -0.05, 0.03];
    let p_init = eye6(0.05);
    let psd = [[1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]];
    let readings = [
        0.01, -0.02, 0.005, // gyro
        0.99, 0.02, -0.01, // magnetometer
        0.999, 0.01, -0.005, 0.002, // star tracker
        0.995, // coarse sun
        0.01, -0.01, // fine sun
        -0.005, 0.01, // fine earth
    ];

    for alpha in [1.0, 0.5] {
        let (qp_sr, wp_sr, pp_sr) = predict(q_init, w_init, p_init, psd, 0.1, alpha, 2.0, 0.0).unwrap();
        let (qp_ukf, wp_ukf, pp_ukf) = ukf_predict(q_init, w_init, p_init, psd, 0.1, alpha, 2.0, 0.0).unwrap();
        for i in 0..4 {
            assert!((qp_sr[i] - qp_ukf[i]).abs() < 1e-11);
        }
        for i in 0..3 {
            assert!((wp_sr[i] - wp_ukf[i]).abs() < 1e-11);
        }
        for i in 0..6 {
            for j in 0..6 {
                assert!((pp_sr[i][j] - pp_ukf[i][j]).abs() < 1e-10);
            }
        }

        let (qs_sr, ws_sr, ps_sr) =
            step(q_init, w_init, p_init, psd, 0.1, &readings, &config, alpha, 2.0, 0.0).unwrap();
        let (qs_ukf, ws_ukf, ps_ukf) =
            ukf_step(q_init, w_init, p_init, psd, 0.1, &readings, &config, alpha, 2.0, 0.0).unwrap();
        for i in 0..4 {
            assert!((qs_sr[i] - qs_ukf[i]).abs() < 1e-10);
        }
        for i in 0..3 {
            assert!((ws_sr[i] - ws_ukf[i]).abs() < 1e-10);
        }
        for i in 0..6 {
            for j in 0..6 {
                assert!((ps_sr[i][j] - ps_ukf[i][j]).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn srukf_step_diagnostics_populates_96_element_layout_and_nees() {
    let config = SensorConfig {
        code: vec![0, 4],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        bias: vec![[0.0; 6]; 2],
        noise_cholesky: vec![eye6(1e-3); 2],
        normal: vec![[0.0, 0.0, 1.0]; 2],
        fov: vec![[0.0; 2], [1.0, 1.0]],
        scalars: vec![[0.0; 3]; 2],
        enabled: vec![true; 2],
        offsets: vec![0, 3],
        reference: vec![[0.0; 3], [0.0, 0.0, 1.0]],
    };
    let mut srukf = AttitudeSequentialSRUKF::new(make_base(), 0.5, 2.0, 0.0).unwrap();
    let mut ukf = AttitudeSequentialUKF::new(make_base(), 0.5, 2.0, 0.0).unwrap();
    let truth = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let diag = step_diagnostics(
        &mut srukf,
        0.05,
        &[0.0, 0.0, 0.0, 0.0, 0.0],
        &config,
        &truth,
        true,
    )
    .unwrap();
    let diag_ukf = ukf_step_diagnostics(
        &mut ukf,
        0.05,
        &[0.0, 0.0, 0.0, 0.0, 0.0],
        &config,
        &truth,
        true,
    )
    .unwrap();

    assert!(diag[..4].iter().all(|x| x.is_finite()));
    assert!(diag[4..7].iter().all(|x| x.is_finite()));
    assert!(diag[7..43].iter().all(|x| x.is_finite()));
    assert!(diag[43..45].iter().all(|x| x.is_finite()));
    assert!(diag[45].is_nan()); // width 2 for fine sun
    assert!(diag[55].is_finite()); // NIS
    assert!(diag[56].is_finite()); // NEES
    assert!(diag[57..59].iter().all(|x| x.is_finite()));
    assert!(diag[59].is_nan());
    assert!(diag[78..96].iter().all(|x| x.is_finite()));

    for i in 0..96 {
        if diag_ukf[i].is_nan() {
            assert!(diag[i].is_nan());
        } else {
            assert!((diag[i] - diag_ukf[i]).abs() < 1e-8, "mismatch at index {i}");
        }
    }
}

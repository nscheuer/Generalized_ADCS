use super::attitude_batched_ukf::{
    predict, step, step_diagnostics, AttitudeBatchedUKF, COVARIANCE_SIZE,
};
use super::attitude_mekf::SensorConfig;
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

#[test]
fn batched_ukf_validates_hyperparameters_and_exposes_params() {
    let base = make_base();
    assert_eq!(AttitudeBatchedUKF::COVARIANCE_SIZE, 6);
    assert_eq!(COVARIANCE_SIZE, 6);

    let ukf = AttitudeBatchedUKF::new(base.clone(), 0.5, 2.0, 1.0).unwrap();
    assert_eq!(ukf.ukf_params(), [0.5, 2.0, 1.0]);

    assert!(AttitudeBatchedUKF::new(base.clone(), 0.0, 2.0, 0.0).is_err());
    assert!(AttitudeBatchedUKF::new(base.clone(), -0.5, 2.0, 0.0).is_err());
    assert!(AttitudeBatchedUKF::new(base.clone(), f64::NAN, 2.0, 0.0).is_err());
    assert!(AttitudeBatchedUKF::new(base.clone(), 1.0, f64::INFINITY, 0.0).is_err());
    assert!(AttitudeBatchedUKF::new(base.clone(), 1.0, 2.0, f64::NAN).is_err());
    assert!(AttitudeBatchedUKF::new(base, 1.0, 2.0, -6.0).is_err());
}

#[test]
fn batched_ukf_predict_propagates_state_and_process_noise() {
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
fn batched_ukf_step_is_order_invariant() {
    let config_a = SensorConfig {
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
    };
    let config_b = SensorConfig {
        code: vec![5, 4, 3, 2, 1, 0],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        bias: vec![[0.0; 6]; 6],
        noise_cholesky: vec![eye6(1e-3); 6],
        normal: vec![[0.0, 0.0, 1.0]; 6],
        fov: vec![
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
        ],
        scalars: vec![
            [0.0; 3],
            [0.0; 3],
            [10.0, 1.0, 1e-3],
            [0.0; 3],
            [10.0, 0.0, 0.0],
            [0.0; 3],
        ],
        enabled: vec![true; 6],
        offsets: vec![0, 2, 4, 5, 9, 12],
        reference: vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0; 3],
        ],
    };
    let q_init = [0.05_f64.cos(), 0.05_f64.sin(), 0.0, 0.0];
    let w_init = [0.1, -0.1, 0.05];
    let readings_a = [
        0.01, -0.02, 0.03, // gyro
        1.0, 0.02, -0.01, // magnetometer
        1.0, 0.0, 0.0, 0.0, // star tracker
        0.99, // coarse sun
        0.01, -0.01, // fine sun
        -0.01, 0.02, // fine earth
    ];
    let readings_b = [
        -0.01, 0.02, // fine earth
        0.01, -0.01, // fine sun
        0.99, // coarse sun
        1.0, 0.0, 0.0, 0.0, // star tracker
        1.0, 0.02, -0.01, // magnetometer
        0.01, -0.02, 0.03, // gyro
    ];
    let psd = [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]];
    let (qa, wa, pa) = step(
        q_init, w_init, eye6(0.1), psd, 0.1, &readings_a, &config_a, 1.0, 2.0, 0.0,
    )
    .unwrap();
    let (qb, wb, pb) = step(
        q_init, w_init, eye6(0.1), psd, 0.1, &readings_b, &config_b, 1.0, 2.0, 0.0,
    )
    .unwrap();
    for i in 0..4 {
        assert!((qa[i] - qb[i]).abs() < 1e-10, "q[{i}] mismatch: {} vs {}", qa[i], qb[i]);
    }
    for i in 0..3 {
        assert!((wa[i] - wb[i]).abs() < 1e-10, "w[{i}] mismatch: {} vs {}", wa[i], wb[i]);
    }
    for i in 0..6 {
        for j in 0..6 {
            assert!((pa[i][j] - pb[i][j]).abs() < 1e-10, "p[{i}][{j}] mismatch: {} vs {}", pa[i][j], pb[i][j]);
        }
    }
}

#[test]
fn batched_ukf_step_consumes_all_six_sensor_types_and_skips_nan_and_fov_dropout() {
    let config = SensorConfig {
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
    };
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
fn batched_ukf_step_diagnostics_populates_96_element_layout_and_nees() {
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
    let mut ukf = AttitudeBatchedUKF::new(make_base(), 0.5, 2.0, 0.0).unwrap();
    let truth = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let diag = step_diagnostics(
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
    assert!(diag[43..46].iter().all(|x| x.is_finite())); // m_total = 5 -> diag_w = 3
    assert!(diag[46..55].iter().all(|x| x.is_finite())); // p_zz 3x3
    assert!(diag[55].is_finite()); // NIS
    assert!(diag[56].is_finite()); // NEES
    assert!(diag[57..60].iter().all(|x| x.is_finite())); // post_res
    assert!(diag[60..78].iter().all(|x| x.is_finite())); // gain 6x3
    assert!(diag[78..96].iter().all(|x| x.is_finite()));
}

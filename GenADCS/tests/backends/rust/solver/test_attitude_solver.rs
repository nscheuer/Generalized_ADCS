use crate::attitude_solver::AttitudeSolverConfig;

#[test]
fn accepts_positive_finite_dt_and_rejects_invalid() {
    let cfg = AttitudeSolverConfig::new(0.25).unwrap();
    assert!((cfg.dt - 0.25).abs() < 1e-15);

    for bad in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(AttitudeSolverConfig::new(bad).is_err());
    }
}

use crate::orbital_solver::OrbitalSolverConfig;

#[test]
fn accepts_positive_finite_dt_and_rejects_invalid() {
    let cfg = OrbitalSolverConfig::new(12.5).unwrap();
    assert!((cfg.dt - 12.5).abs() < 1e-15);

    for bad in [0.0, -0.5, f64::INFINITY, f64::NAN] {
        assert!(OrbitalSolverConfig::new(bad).is_err());
    }
}

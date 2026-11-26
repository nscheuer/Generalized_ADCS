use crate::coupled_solver::CoupledSolverConfig;

#[test]
fn accepts_valid_configuration_and_zero_duration() {
    let cfg = CoupledSolverConfig::new(10.0, true, 20).unwrap();
    assert_eq!(cfg.final_time, 10.0);
    assert!(cfg.progress);
    assert_eq!(cfg.progress_chunk_steps, 20);

    let zero = CoupledSolverConfig::new(0.0, false, 500).unwrap();
    assert_eq!(zero.final_time, 0.0);
    assert!(!zero.progress);
    assert_eq!(zero.progress_chunk_steps, 500);
}

#[test]
fn rejects_invalid_final_time_or_chunk_steps() {
    assert!(CoupledSolverConfig::new(-1.0, false, 500).is_err());
    assert!(CoupledSolverConfig::new(f64::INFINITY, false, 500).is_err());
    assert!(CoupledSolverConfig::new(f64::NAN, false, 500).is_err());
    assert!(CoupledSolverConfig::new(1.0, false, 0).is_err());
}

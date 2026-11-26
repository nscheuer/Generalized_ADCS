use crate::solver::{SolverConfig, SolverMode};

#[test]
fn accepts_split_and_coupled_compositions() {
    let split = SolverConfig::new(false, true, true).unwrap();
    assert_eq!(split.mode, SolverMode::Split);

    let coupled = SolverConfig::new(true, false, false).unwrap();
    assert_eq!(coupled.mode, SolverMode::Coupled);
}

#[test]
fn rejects_incomplete_or_ambiguous_compositions() {
    for (coupled, orbital, attitude) in [
        (false, false, false),
        (false, true, false),
        (false, false, true),
        (true, true, true),
        (true, true, false),
        (true, false, true),
    ] {
        assert!(SolverConfig::new(coupled, orbital, attitude).is_err());
    }
}

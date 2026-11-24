/// Solver composition selected by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverMode {
    Split,
    Coupled,
}

/// Rust-native validation and description of a solver composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolverConfig {
    pub mode: SolverMode,
}

impl SolverConfig {
    pub fn new(has_coupled: bool, has_orbital: bool, has_attitude: bool) -> Result<Self, String> {
        match (has_coupled, has_orbital, has_attitude) {
            (true, false, false) => Ok(Self {
                mode: SolverMode::Coupled,
            }),
            (false, true, true) => Ok(Self {
                mode: SolverMode::Split,
            }),
            _ => Err("provide a coupled solver or both orbital and attitude solvers".to_owned()),
        }
    }
}

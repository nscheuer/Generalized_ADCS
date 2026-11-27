use super::fdir::FDIR;
use super::state_machine::{StateMachine, TimelineEntry};

#[derive(Clone)]
pub struct OBCSubsystem {
    pub state_machine: Option<StateMachine>,
    pub fdir: Option<FDIR>,
    #[allow(non_snake_case)]
    pub N: usize,
    pub timeline: Vec<TimelineEntry>,
}

impl Default for OBCSubsystem {
    fn default() -> Self {
        Self {
            state_machine: None,
            fdir: None,
            N: 1,
            timeline: Vec::new(),
        }
    }
}

impl OBCSubsystem {
    pub fn new(
        state_machine: Option<StateMachine>,
        fdir: Option<FDIR>,
        n: usize,
        timeline: Vec<TimelineEntry>,
    ) -> Result<Self, String> {
        if n < 1 {
            return Err("N must be a positive integer".into());
        }
        for (t, cmd, _) in &timeline {
            if !t.is_finite() || *t < 0.0 {
                return Err("timeline times must be finite and nonnegative".into());
            }
            if cmd.is_empty() {
                return Err("timeline telecommand names must be nonempty".into());
            }
        }
        Ok(Self {
            state_machine,
            fdir,
            N: n,
            timeline,
        })
    }
}

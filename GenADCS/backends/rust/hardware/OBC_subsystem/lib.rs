#![allow(non_local_definitions)]

use pyo3::prelude::*;

pub mod fdir;
pub mod obc_py;
pub mod obc_subsystem;
pub mod runtime;
pub mod state_machine;

pub use fdir::{AngularRateMonitor, FDIRMonitor, FDIR};
pub use obc_subsystem::OBCSubsystem;
pub use runtime::{init_obc_work, prepare_obc, step_obc, step_obc_core, GoalTable, OBCRuntime};
pub use state_machine::{
    ADCSBinding, After, Completion, FDIRTriggered, HardwareMaskSpec, InFOV, Mode, ModeBindings,
    StateMachine, Telecommand, TimelineEntry, Transition, Trigger,
};

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/OBC_subsystem/test_obc_and_adcs_state_machine.rs"]
mod test_obc_and_adcs_state_machine;

#[pymodule]
fn _obc_subsystem(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    obc_py::register(m)
}

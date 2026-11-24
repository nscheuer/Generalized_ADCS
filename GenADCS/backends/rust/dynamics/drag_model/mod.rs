pub mod projected_area;
#[path = "projected_area_py.rs"]
mod python;
pub use python::{projected_area_drag_force, projected_area_drag_torque};

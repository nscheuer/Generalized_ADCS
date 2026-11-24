#[path = "static_py.rs"]
mod python;
#[path = "static.rs"]
pub mod static_model;
pub use python::static_gravity_gradient_torque;

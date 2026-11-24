/// Native summary of configured force and torque model kinds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamicsConfigCore {
    pub force_model_kinds: Vec<String>,
    pub torque_model_kinds: Vec<String>,
}

impl DynamicsConfigCore {
    pub fn new(force_model_kinds: Vec<String>, torque_model_kinds: Vec<String>) -> Self {
        Self {
            force_model_kinds,
            torque_model_kinds,
        }
    }
}

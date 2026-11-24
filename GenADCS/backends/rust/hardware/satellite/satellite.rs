/// Satellite configuration containing geometry, ADCS, and state-machine values.
///
/// The type is generic so Rust callers can use native types while language
/// adapters can store their own handles without coupling this module to them.
pub struct Satellite<G, A, S> {
    pub geometry: G,
    pub adcs: A,
    pub state_machine: S,
}

impl<G, A, S> Satellite<G, A, S> {
    pub fn new(geometry: G, adcs: A, state_machine: S) -> Self {
        Self {
            geometry,
            adcs,
            state_machine,
        }
    }

    pub fn into_parts(self) -> (G, A, S) {
        (self.geometry, self.adcs, self.state_machine)
    }
}

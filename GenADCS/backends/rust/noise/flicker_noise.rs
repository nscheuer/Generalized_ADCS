/// Configuration for flicker noise.
///
/// The Python implementation is a dataclass that stores its amplitude; the
/// simulation process itself is handled by the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlickerNoise {
    pub amplitude: f64,
}

impl FlickerNoise {
    pub fn new(amplitude: f64) -> Self {
        Self { amplitude }
    }
}

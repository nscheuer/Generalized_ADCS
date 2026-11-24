/// Rust-native DE421 ephemeris backed by an SPK kernel.
pub struct De421 {
    kernel: Kernel,
}

/// Position and velocity in the kernel's native units (km and km/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct De421State {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

impl De421 {
    /// Load a DE421 kernel from a file path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        Kernel::load(path.as_ref())
            .map(|kernel| Self { kernel })
            .map_err(|error| error.to_string())
    }

    /// Return the available body names.
    pub fn available_bodies() -> &'static [&'static str] {
        &BODY_NAMES
    }

    /// Return a body's state relative to Earth at a Julian date in TDB.
    /// Body identifiers follow the DE421 table: Mercury 0 through Moon 9, Sun 10, Earth 11.
    pub fn state_eci(&self, body_index: usize, epoch_jd_tdb: f64) -> Result<De421State, String> {
        let state = self
            .kernel
            .planetary_state(body_index, epoch_jd_tdb)
            .map_err(|error| error.to_string())?;
        Ok(De421State {
            position: state.position,
            velocity: state.velocity,
        })
    }

    /// Return a body's position relative to Earth at a Julian date in TDB, in km.
    pub fn position_eci(&self, body_index: usize, epoch_jd_tdb: f64) -> Result<[f64; 3], String> {
        self.state_eci(body_index, epoch_jd_tdb).map(|state| state.position)
    }

    /// Return a body's velocity relative to Earth at a Julian date in TDB, in km/s.
    pub fn velocity_eci(&self, body_index: usize, epoch_jd_tdb: f64) -> Result<[f64; 3], String> {
        self.state_eci(body_index, epoch_jd_tdb).map(|state| state.velocity)
    }
}

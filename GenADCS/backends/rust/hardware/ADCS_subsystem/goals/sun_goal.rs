use std::path::{Path, PathBuf};
use std::sync::Arc;

use ephemeris_native::NativeEphemeris;

use super::goal::{Goal, GoalReference};
use super::orbit_goal_base::{orbit_reference, OrbitMode};

/// Default workspace path to the bundled DE421 SPK kernel.
pub fn default_de421_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../kernels/environment/ephemeris/de421/de421.bsp")
}

/// Point body +Z toward the Sun using the configured DE421 ephemeris.
pub struct SunGoal {
    ephemeris: Arc<NativeEphemeris>,
}

impl SunGoal {
    pub fn new() -> Result<Self, String> {
        Self::from_path(&default_de421_path())
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let ephemeris = Arc::new(NativeEphemeris::from_path(path)?);
        Ok(Self { ephemeris })
    }

    pub fn from_ephemeris(ephemeris: Arc<NativeEphemeris>) -> Self {
        Self { ephemeris }
    }
}

impl Goal for SunGoal {
    fn reference(&self, state: &[f64], epoch_jd: f64, mu: f64) -> Result<GoalReference, String> {
        orbit_reference(state, epoch_jd, mu, OrbitMode::Sun, Some(&self.ephemeris))
    }
}

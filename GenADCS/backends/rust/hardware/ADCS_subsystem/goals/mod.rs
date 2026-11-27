#[path = "_orbit_goal.rs"]
mod orbit_goal_base;

pub mod along_track_goal;
pub mod angular_rate_goal;
pub mod coordinate_fmc_goal;
pub mod coordinate_goal;
pub mod cross_track_goal;
pub mod goal;
pub mod goals_py;
pub mod nadir_goal;
pub mod no_goal;
pub mod quaternion_goal;
pub mod sun_goal;
pub mod two_vector_goal;
pub mod vector_goal;
pub mod velocity_goal;
pub mod zenith_goal;

pub use along_track_goal::AlongTrackGoal;
pub use angular_rate_goal::AngularRateGoal;
pub use coordinate_fmc_goal::CoordinateFMCGoal;
pub use coordinate_goal::CoordinateGoal;
pub use cross_track_goal::CrossTrackGoal;
pub use goal::{
    validate_quaternion, validate_reference_state, validate_vector, Goal, GoalReference,
    DEFAULT_MU,
};
pub use nadir_goal::NadirGoal;
pub use no_goal::NoGoal;
pub use quaternion_goal::QuaternionGoal;
pub use sun_goal::SunGoal;
pub use two_vector_goal::TwoVectorGoal;
pub use vector_goal::VectorGoal;
pub use velocity_goal::VelocityGoal;
pub use zenith_goal::ZenithGoal;

#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/goals/test_goals.rs"]
mod test_goals;

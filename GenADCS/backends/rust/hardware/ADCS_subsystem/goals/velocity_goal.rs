use super::orbit_goal_base::{define_orbit_goal, OrbitMode};

define_orbit_goal!(
    VelocityGoal,
    OrbitMode::Velocity,
    "Point body +Z along instantaneous velocity; +X follows radial."
);

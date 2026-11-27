use super::orbit_goal_base::{define_orbit_goal, OrbitMode};

define_orbit_goal!(
    ZenithGoal,
    OrbitMode::Zenith,
    "Point body +Z away from the central body; +X follows orbit tangent."
);

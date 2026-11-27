use super::orbit_goal_base::{define_orbit_goal, OrbitMode};

define_orbit_goal!(
    NadirGoal,
    OrbitMode::Nadir,
    "Point body +Z toward the central body; +X follows orbit tangent."
);

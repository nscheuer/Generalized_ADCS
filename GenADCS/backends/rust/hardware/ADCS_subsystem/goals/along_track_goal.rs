use super::orbit_goal_base::{define_orbit_goal, OrbitMode};

define_orbit_goal!(
    AlongTrackGoal,
    OrbitMode::AlongTrack,
    "Point body +Z along orbit tangent; +X follows radial."
);

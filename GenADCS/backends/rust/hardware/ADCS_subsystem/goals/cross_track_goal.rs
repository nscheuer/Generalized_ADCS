use super::orbit_goal_base::{define_orbit_goal, OrbitMode};

define_orbit_goal!(
    CrossTrackGoal,
    OrbitMode::CrossTrack,
    "Point body +Z along orbit normal; +X follows radial."
);

#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_CROSS_TRACK_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_CROSS_TRACK_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class CrossTrackGoal final : public OrbitGoal {
public:
  CrossTrackGoal() : OrbitGoal(OrbitGoalMode::CrossTrack) {}
};

} // namespace omni::hardware::goals

#endif

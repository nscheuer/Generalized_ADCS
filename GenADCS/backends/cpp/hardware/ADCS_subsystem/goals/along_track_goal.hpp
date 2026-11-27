#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ALONG_TRACK_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ALONG_TRACK_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class AlongTrackGoal final : public OrbitGoal {
public:
  AlongTrackGoal() : OrbitGoal(OrbitGoalMode::AlongTrack) {}
};

} // namespace omni::hardware::goals

#endif

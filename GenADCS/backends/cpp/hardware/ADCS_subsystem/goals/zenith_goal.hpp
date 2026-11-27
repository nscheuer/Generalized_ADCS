#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ZENITH_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ZENITH_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class ZenithGoal final : public OrbitGoal {
public:
  ZenithGoal() : OrbitGoal(OrbitGoalMode::Zenith) {}
};

} // namespace omni::hardware::goals

#endif

#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_SUN_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_SUN_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class SunGoal final : public OrbitGoal {
public:
  SunGoal() : OrbitGoal(OrbitGoalMode::Sun) {}
};

} // namespace omni::hardware::goals

#endif

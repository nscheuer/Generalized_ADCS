#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_VELOCITY_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_VELOCITY_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class VelocityGoal final : public OrbitGoal {
public:
  VelocityGoal() : OrbitGoal(OrbitGoalMode::Velocity) {}
};

} // namespace omni::hardware::goals

#endif

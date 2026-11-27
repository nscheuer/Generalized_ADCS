#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_NADIR_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_NADIR_GOAL_HPP

#include "_orbit_goal.hpp"

namespace omni::hardware::goals {

class NadirGoal final : public OrbitGoal {
public:
  NadirGoal() : OrbitGoal(OrbitGoalMode::Nadir) {}
};

} // namespace omni::hardware::goals

#endif

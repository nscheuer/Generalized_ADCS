#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ORBIT_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ORBIT_GOAL_HPP

#include "goal.hpp"

namespace omni::hardware::goals {

enum class OrbitGoalMode {
  Nadir = 0,
  Zenith = 1,
  AlongTrack = 2,
  CrossTrack = 3,
  Velocity = 4,
  Sun = 5,
};

class OrbitGoal : public Goal {
public:
  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

protected:
  explicit OrbitGoal(OrbitGoalMode mode) : mode_(mode) {}

private:
  OrbitGoalMode mode_;
};

} // namespace omni::hardware::goals

#endif

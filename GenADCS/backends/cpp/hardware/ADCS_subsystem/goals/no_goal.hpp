#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_NO_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_NO_GOAL_HPP

#include "goal.hpp"

namespace omni::hardware::goals {

class NoGoal final : public Goal {
public:
  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;
};

} // namespace omni::hardware::goals

#endif

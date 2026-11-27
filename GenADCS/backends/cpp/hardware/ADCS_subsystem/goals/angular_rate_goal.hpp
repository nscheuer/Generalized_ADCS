#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ANGULAR_RATE_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_ANGULAR_RATE_GOAL_HPP

#include "goal.hpp"

#include <Eigen/Core>

namespace omni::hardware::goals {

class AngularRateGoal final : public Goal {
public:
  explicit AngularRateGoal(const Eigen::Ref<const Eigen::Vector3d> &rate);

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  const Vector3 &rate() const { return rate_; }

private:
  Vector3 rate_;
};

} // namespace omni::hardware::goals

#endif

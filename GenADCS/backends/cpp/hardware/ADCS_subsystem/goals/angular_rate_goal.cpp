#include "angular_rate_goal.hpp"

#include <stdexcept>

namespace omni::hardware::goals {

AngularRateGoal::AngularRateGoal(const Eigen::Ref<const Eigen::Vector3d> &rate)
    : rate_(rate) {
  if (!rate_.allFinite()) {
    throw std::invalid_argument("rate must be a finite 3-element vector");
  }
}

GoalReference AngularRateGoal::reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double /*epoch_jd*/,
    const GoalRuntime & /*runtime*/) const {
  validate_reference_state(state);
  return {std::nullopt, rate_};
}

} // namespace omni::hardware::goals

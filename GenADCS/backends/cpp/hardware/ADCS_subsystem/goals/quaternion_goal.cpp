#include "quaternion_goal.hpp"

#include <stdexcept>

namespace omni::hardware::goals {

QuaternionGoal::QuaternionGoal(const Eigen::Ref<const Eigen::Vector4d> &quaternion,
                               const std::optional<Eigen::Vector3d> &rate)
    : quaternion_(validate_quaternion(quaternion)),
      rate_(rate.has_value() ? *rate : Vector3::Zero()) {
  if (!rate_.allFinite()) {
    throw std::invalid_argument("rate must be a finite 3-element vector");
  }
}

GoalReference QuaternionGoal::reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double /*epoch_jd*/,
    const GoalRuntime & /*runtime*/) const {
  validate_reference_state(state);
  return {quaternion_, rate_};
}

} // namespace omni::hardware::goals

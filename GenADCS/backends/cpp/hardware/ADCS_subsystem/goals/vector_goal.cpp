#include "vector_goal.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::goals {

VectorGoal::VectorGoal(const Eigen::Ref<const Eigen::Vector3d> &inertial_vector,
                       const Eigen::Ref<const Eigen::Vector3d> &body_axis,
                       const Eigen::Ref<const Eigen::Vector3d> &inertial_roll,
                       const Eigen::Ref<const Eigen::Vector3d> &body_roll)
    : inertial_vector_(validate_vector(inertial_vector, "inertial_vector")),
      body_axis_(validate_vector(body_axis, "body_axis")),
      inertial_roll_(validate_vector(inertial_roll, "inertial_roll")),
      body_roll_(validate_vector(body_roll, "body_roll")) {
  if (std::abs(inertial_vector_.dot(inertial_roll_)) > 1.0 - 1e-10) {
    throw std::invalid_argument(
        "inertial_roll must not be parallel to inertial_vector");
  }
  if (std::abs(body_axis_.dot(body_roll_)) > 1.0 - 1e-10) {
    throw std::invalid_argument("body_roll must not be parallel to body_axis");
  }
}

GoalReference VectorGoal::reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                                    double /*epoch_jd*/,
                                    const GoalRuntime & /*runtime*/) const {
  validate_reference_state(state);
  return {detail::triad_quaternion(inertial_vector_, inertial_roll_, body_axis_,
                                   body_roll_),
          Vector3::Zero()};
}

} // namespace omni::hardware::goals

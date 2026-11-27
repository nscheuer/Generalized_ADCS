#include "two_vector_goal.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::goals {

TwoVectorGoal::TwoVectorGoal(const Eigen::Matrix<double, 2, 3> &inertial_vectors,
                             const Eigen::Matrix<double, 2, 3> &body_vectors)
    : inertial_first_(
          validate_vector(inertial_vectors.row(0), "inertial_vectors[0]")),
      inertial_second_(
          validate_vector(inertial_vectors.row(1), "inertial_vectors[1]")),
      body_first_(validate_vector(body_vectors.row(0), "body_vectors[0]")),
      body_second_(validate_vector(body_vectors.row(1), "body_vectors[1]")) {
  if (std::abs(inertial_first_.dot(inertial_second_)) > 1.0 - 1e-10) {
    throw std::invalid_argument("inertial vectors must not be parallel");
  }
  if (std::abs(body_first_.dot(body_second_)) > 1.0 - 1e-10) {
    throw std::invalid_argument("body vectors must not be parallel");
  }
}

GoalReference TwoVectorGoal::reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double /*epoch_jd*/,
    const GoalRuntime & /*runtime*/) const {
  validate_reference_state(state);
  return {detail::triad_quaternion(inertial_first_, inertial_second_,
                                   body_first_, body_second_),
          Vector3::Zero()};
}

} // namespace omni::hardware::goals

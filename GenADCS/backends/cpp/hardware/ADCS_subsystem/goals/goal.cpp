#include "goal.hpp"

#include "helpers/math/rotations/rotation_types.hpp"

#include <Eigen/Geometry>
#include <stdexcept>

namespace omni::hardware::goals {

void validate_reference_state(const Eigen::Ref<const Eigen::VectorXd> &state) {
  if (state.size() < 13 || !state.allFinite()) {
    throw std::invalid_argument(
        "goal state must have a finite 13-element spacecraft-state prefix");
  }
  if (state.head<4>().norm() == 0.0) {
    throw std::invalid_argument("goal state quaternion must be nonzero");
  }
}

Quaternion validate_quaternion(const Eigen::Ref<const Eigen::Vector4d> &quaternion) {
  if (!quaternion.allFinite()) {
    throw std::invalid_argument("quaternion must be a finite 4-element vector");
  }
  const double norm = quaternion.norm();
  if (norm == 0.0) {
    throw std::invalid_argument("quaternion must be nonzero");
  }
  return quaternion / norm;
}

Vector3 validate_vector(const Eigen::Ref<const Eigen::Vector3d> &vector,
                        const std::string &name) {
  if (!vector.allFinite()) {
    throw std::invalid_argument(name + " must be a finite 3-element vector");
  }
  const double norm = vector.norm();
  if (norm == 0.0) {
    throw std::invalid_argument(name + " must be nonzero");
  }
  return vector / norm;
}

namespace detail {

Quaternion triad_quaternion(const Vector3 &i0, const Vector3 &i_roll,
                            const Vector3 &b0, const Vector3 &b_roll) {
  Vector3 i1 = i_roll - i_roll.dot(i0) * i0;
  i1.normalize();
  const Vector3 i2 = i0.cross(i1);

  Vector3 b1 = b_roll - b_roll.dot(b0) * b0;
  b1.normalize();
  const Vector3 b2 = b0.cross(b1);

  Eigen::Matrix3d inertial_basis;
  inertial_basis.col(0) = i0;
  inertial_basis.col(1) = i1;
  inertial_basis.col(2) = i2;

  Eigen::Matrix3d body_basis;
  body_basis.col(0) = b0;
  body_basis.col(1) = b1;
  body_basis.col(2) = b2;

  return omni::rotations::dcm_to_quaternion(body_basis *
                                            inertial_basis.transpose());
}

} // namespace detail

} // namespace omni::hardware::goals

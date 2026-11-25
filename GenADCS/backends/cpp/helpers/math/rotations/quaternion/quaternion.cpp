#include "quaternion.hpp"
#include "../../vector.hpp"
#include "algebra.hpp"
#include "conversions.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/LU>

namespace omni::rotations {
namespace {
void require_proper_dcm(const Matrix3 &matrix) {
  if (!matrix.allFinite()) {
    throw std::invalid_argument("DCM must be finite");
}
  const Matrix3 residual = matrix.transpose() * matrix - Matrix3::Identity();
  if ((residual.norm() > 1e-8) ||
      (std::abs(matrix.determinant() - 1.0) > 1e-8)) {
    throw std::invalid_argument("DCM must be a proper rotation matrix");
  }
}
std::string convention_key(const std::string &convention) {
  std::string key = convention;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return key;
}
} // namespace

Quaternion::Quaternion(const QuaternionVector &values,
                       const std::string &convention) {
  const std::string key = convention_key(convention);
  if ((key == "hamilton") || (key == "h")) {
    hamilton_ = normalize_quaternion(values);
    convention_ = "Hamilton";
  } else if ((key == "jpl") || (key == "j")) {
    hamilton_ = normalize_quaternion(jpl_to_hamilton(values));
    convention_ = "JPL";
  } else {
    throw std::invalid_argument("convention must be 'Hamilton' or 'JPL'");
  }
}
Quaternion Quaternion::identity() { return Quaternion(); }
Quaternion Quaternion::from_axis_angle(const Vector3 &axis, double angle) {
  return Quaternion(axis_angle_to_quaternion(axis, angle));
}
Quaternion Quaternion::from_rotation_vector(const Vector3 &vector) {
  return Quaternion(rotation_vector_to_quaternion(vector));
}
Quaternion Quaternion::from_dcm(const Matrix3 &matrix) {
  require_proper_dcm(matrix);
  return Quaternion(dcm_to_quaternion(matrix));
}
QuaternionVector Quaternion::components() const {
  return as_convention(convention_);
}
QuaternionVector Quaternion::hamilton_components() const { return hamilton_; }
std::string Quaternion::convention() const { return convention_; }
double Quaternion::norm() const { return quaternion_norm(hamilton_); }
QuaternionVector
Quaternion::as_convention(const std::string &convention) const {
  const std::string key = convention_key(convention);
  if ((key == "hamilton") || (key == "h")) {
    return hamilton_;
}
  if ((key == "jpl") || (key == "j")) {
    return hamilton_to_jpl(hamilton_);
}
  throw std::invalid_argument("convention must be 'Hamilton' or 'JPL'");
}
Quaternion Quaternion::to_convention(const std::string &convention) const {
  return Quaternion(as_convention(convention), convention);
}
Quaternion Quaternion::normalized() const {
  return Quaternion(normalize_quaternion(hamilton_));
}
Quaternion Quaternion::conjugate() const {
  return Quaternion(rotations::conjugate(hamilton_));
}
Quaternion Quaternion::inverse() const {
  return Quaternion(rotations::inverse(hamilton_));
}
Quaternion Quaternion::multiply(const Quaternion &other) const {
  return Quaternion(rotations::multiply(hamilton_, other.hamilton_));
}
Vector3 Quaternion::rotate(const Vector3 &vector) const {
  return rotate_vector(hamilton_, vector);
}
Vector3 Quaternion::inertial_to_body(const Vector3 &vector) const {
  return rotate(vector);
}
Vector3 Quaternion::body_to_inertial(const Vector3 &vector) const {
  return rotate_vector(rotations::conjugate(hamilton_), vector);
}
Matrix3 Quaternion::to_dcm() const { return quaternion_to_dcm(hamilton_); }
Quaternion Quaternion::relative_to(const Quaternion &reference) const {
  return Quaternion(relative_quaternion(reference.hamilton_, hamilton_));
}
} // namespace omni::rotations

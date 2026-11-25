#include "base.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::estimator {
namespace {
template <typename Matrix>
void validate_psd(const Matrix &value, const char *name) {
  if (!value.allFinite() || !value.isApprox(value.transpose(), 1e-8) ||
      Eigen::SelfAdjointEigenSolver<Matrix>(value).eigenvalues().minCoeff() <
          -1e-12) {
    throw std::invalid_argument(
        std::string(name) +
        " must be finite, symmetric and positive semidefinite");
}
}
std::set<std::string> validate_record(std::set<std::string> record) {
  static const std::set<std::string> allowed = {
      "estimated_attitude",
      "estimated_rate",
      "estimated_attitude_covariance",
      "estimated_rate_covariance",
      "measurement_innovation",
      "measurement_innovation_covariance",
      "normalized_innovation_squared",
      "normalized_estimation_error_squared",
      "measurement_residual",
      "kalman_gain"};
  for (const auto &field : record) {
    if (!allowed.count(field)) {
      throw std::invalid_argument("unknown estimator diagnostic field: " +
                                  field);
}
}
  return record;
}
Eigen::Vector4d normalize(Eigen::Vector4d q) {
  if (!q.allFinite() || q.norm() == 0.0) {
    throw std::invalid_argument("quaternion must be finite and nonzero");
}
  return q.normalized();
}
Eigen::Vector3d validate_rate(Eigen::Vector3d w) {
  if (!w.allFinite()) {
    throw std::invalid_argument("rate must be finite");
}
  return w;
}
Matrix6 validate_covariance(Matrix6 p) {
  validate_psd(p, "covariance");
  return p;
}
Eigen::Matrix3d validate_acceleration(Eigen::Matrix3d a) {
  validate_psd(a, "acceleration_psd");
  return a;
}
} // namespace
AttitudeEstimator::AttitudeEstimator(Eigen::Vector4d q, Eigen::Vector3d w,
                                     Matrix6 p, Eigen::Matrix3d a,
                                     std::set<std::string> r)
    : quaternion(normalize(q)), rate(validate_rate(w)),
      covariance(validate_covariance(p)),
      acceleration_psd(validate_acceleration(a)),
      record(validate_record(std::move(r))) {}
State AttitudeEstimator::prepare() const {
  return {quaternion, rate, covariance};
}
} // namespace omni::hardware::estimator

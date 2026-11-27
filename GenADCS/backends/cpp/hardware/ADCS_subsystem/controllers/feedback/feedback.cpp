#include "feedback.hpp"

#include <Eigen/Eigenvalues>
#include <stdexcept>

namespace omni::hardware::controllers {

namespace {

Eigen::Matrix3d validate_gain_matrix(const Eigen::Matrix3d &mat,
                                     const std::string &name,
                                     bool require_spd) {
  if (!mat.allFinite()) {
    throw std::invalid_argument(name + " must be finite");
  }
  if (!mat.isApprox(mat.transpose(), 1e-8)) {
    throw std::invalid_argument(name + " must be symmetric");
  }
  Eigen::Matrix3d sym = 0.5 * (mat + mat.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(sym);
  if (solver.info() != Eigen::Success) {
    throw std::invalid_argument(name + " eigendecomposition failed");
  }
  const double min_eig = solver.eigenvalues().minCoeff();
  if (require_spd && min_eig <= 0.0) {
    throw std::invalid_argument(name + " must be positive definite");
  }
  if (!require_spd && min_eig < 0.0) {
    throw std::invalid_argument(name + " must be positive semidefinite");
  }
  return mat;
}

} // namespace

Eigen::Matrix3d to_gain_matrix(double value, const std::string &name,
                               bool require_spd) {
  return validate_gain_matrix(value * Eigen::Matrix3d::Identity(), name,
                              require_spd);
}

Eigen::Matrix3d to_gain_matrix(const Eigen::Vector3d &value,
                               const std::string &name, bool require_spd) {
  return validate_gain_matrix(value.asDiagonal(), name, require_spd);
}

Eigen::Matrix3d to_gain_matrix(const Eigen::Matrix3d &value,
                               const std::string &name, bool require_spd) {
  return validate_gain_matrix(value, name, require_spd);
}

} // namespace omni::hardware::controllers

#pragma once
#include <Eigen/Dense>
#include <set>
#include <string>

namespace omni::hardware::estimator {
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Matrix7 = Eigen::Matrix<double, 7, 7>;
struct State {
  Eigen::Vector4d quaternion;
  Eigen::Vector3d rate;
  Eigen::MatrixXd covariance;
};
class AttitudeEstimator {
public:
  AttitudeEstimator(Eigen::Vector4d quaternion, Eigen::Vector3d rate,
                    Eigen::MatrixXd covariance,
                    Eigen::Matrix3d acceleration_psd,
                    std::set<std::string> record = {});
  virtual ~AttitudeEstimator() = default;
  State prepare() const;
  const Eigen::Vector4d quaternion;
  const Eigen::Vector3d rate;
  const Eigen::MatrixXd covariance;
  const Eigen::Matrix3d acceleration_psd;
  const std::set<std::string> record;
};
} // namespace omni::hardware::estimator

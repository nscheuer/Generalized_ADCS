#pragma once
#include "base.hpp"
#include <map>
#include <string>
#include <vector>

namespace omni::hardware::estimator {
struct Sensor {
  int code = -1; // 0 gyro, 2 tracker
  Eigen::Vector4d orientation = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector4d misalignment = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector3d bias = Eigen::Vector3d::Zero();
  Eigen::Matrix3d noise_cholesky = Eigen::Matrix3d::Zero();
  bool enabled = true;
  Eigen::Index offset = 0;
};
struct Correction {
  State state;
  Eigen::Vector3d innovation;
  Eigen::Matrix3d innovation_covariance;
  Eigen::Matrix<double, 6, 3> gain;
  Eigen::Vector3d residual;
};
class AttitudeMEKF : public AttitudeEstimator {
public:
  AttitudeMEKF(Eigen::Vector4d quaternion, Eigen::Vector3d rate,
               Eigen::MatrixXd covariance,
               Eigen::Matrix3d acceleration_psd,
               std::set<std::string> record = {});
  static std::map<std::string, Eigen::MatrixXd>
  sample(const AttitudeMEKF &estimator, const Eigen::VectorXd &times,
         const Eigen::MatrixXd &truth, const Eigen::MatrixXd &readings,
         const std::vector<Sensor> &sensors, double interval,
         const std::set<std::string> &fields);
  static State predict(const State &state,
                       const Eigen::Matrix3d &acceleration_psd, double dt);
  static Correction correct(const State &state, const Eigen::Vector3d &residual,
                            const Eigen::Matrix<double, 3, 6> &h,
                            const Eigen::Matrix3d &covariance);
  static Correction correct_gyro(const State &state,
                                 const Eigen::Vector3d &measured_rate,
                                 const Eigen::Matrix3d &mounting,
                                 const Eigen::Matrix3d &covariance);
  static Correction
  correct_star_tracker(const State &state,
                       const Eigen::Vector4d &measured_quaternion,
                       const Eigen::Vector4d &mounting_quaternion,
                       const Eigen::Matrix3d &covariance);
  static State step(const State &state, const Eigen::Matrix3d &acceleration_psd,
                    double dt, const Eigen::VectorXd &readings,
                    const std::vector<Sensor> &sensors);
};
} // namespace omni::hardware::estimator

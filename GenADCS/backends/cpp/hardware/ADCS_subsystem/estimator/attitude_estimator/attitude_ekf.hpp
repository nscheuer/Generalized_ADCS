#pragma once

#include "attitude_mekf.hpp"

namespace omni::hardware::estimator {
struct EKFCorrection {
  State state;
  Eigen::Vector3d innovation;
  Eigen::Matrix3d innovation_covariance;
  Eigen::Matrix<double, 7, 3> gain;
  Eigen::Vector3d residual;
};

class AttitudeEKF : public AttitudeEstimator {
public:
  AttitudeEKF(Eigen::Vector4d quaternion, Eigen::Vector3d rate,
              Eigen::MatrixXd covariance, Eigen::Matrix3d acceleration_psd,
              std::set<std::string> record = {});

  // Update persistent state and write diagnostics for one sensor tick.
  static std::map<std::string, Eigen::MatrixXd>
  allocate_samples(Eigen::Index rows, const std::set<std::string> &fields);
  static void update_sample(State &current,
      const Eigen::Matrix3d &acceleration_psd, double dt,
      const Eigen::VectorXd &truth, const Eigen::VectorXd &readings,
      const std::vector<Sensor> &sensors,
      std::map<std::string, Eigen::MatrixXd> &output, Eigen::Index row);
  static State predict(const State &state,
                       const Eigen::Matrix3d &acceleration_psd, double dt);
  static EKFCorrection correct(const State &state,
                               const Eigen::Vector3d &residual,
                               const Eigen::Matrix<double, 3, 7> &h,
                               const Eigen::Matrix3d &covariance);
  static EKFCorrection correct_gyro(const State &state,
                                    const Eigen::Vector3d &measured_rate,
                                    const Eigen::Matrix3d &mounting,
                                    const Eigen::Matrix3d &covariance);
  static EKFCorrection
  correct_star_tracker(const State &state,
                       const Eigen::Vector4d &measured_quaternion,
                       const Eigen::Vector4d &mounting_quaternion,
                       const Eigen::Matrix3d &covariance);
  static State step(const State &state, const Eigen::Matrix3d &acceleration_psd,
                    double dt, const Eigen::VectorXd &readings,
                    const std::vector<Sensor> &sensors);
  static std::map<std::string, Eigen::MatrixXd>
  sample(const AttitudeEKF &estimator, const Eigen::VectorXd &times,
         const Eigen::MatrixXd &truth, const Eigen::MatrixXd &readings,
         const std::vector<Sensor> &sensors, double interval,
         const std::set<std::string> &fields);
};
} // namespace omni::hardware::estimator

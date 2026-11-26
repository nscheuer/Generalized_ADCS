#pragma once

#include "attitude_mekf.hpp"

namespace omni::hardware::estimator {

class AttitudeSequentialUKF : public AttitudeEstimator {
public:
  static constexpr int COVARIANCE_SIZE = 6;

  AttitudeSequentialUKF(Eigen::Vector4d quaternion, Eigen::Vector3d rate,
                        Eigen::MatrixXd covariance,
                        Eigen::Matrix3d acceleration_psd, double alpha = 1.0,
                        double beta = 2.0, double kappa = 0.0,
                        std::set<std::string> record = {});

  const double alpha;
  const double beta;
  const double kappa;
  const Eigen::Vector3d ukf_params;

  static std::map<std::string, Eigen::MatrixXd>
  allocate_samples(Eigen::Index rows, const std::set<std::string> &fields);

  static void update_sample(State &current,
                            const Eigen::Matrix3d &acceleration_psd, double dt,
                            const Eigen::VectorXd &truth,
                            const Eigen::VectorXd &readings,
                            const std::vector<Sensor> &sensors,
                            std::map<std::string, Eigen::MatrixXd> &output,
                            Eigen::Index row, double alpha = 1.0,
                            double beta = 2.0, double kappa = 0.0);

  static State predict(const State &state,
                       const Eigen::Matrix3d &acceleration_psd, double dt,
                       double alpha = 1.0, double beta = 2.0,
                       double kappa = 0.0);

  static State step(const State &state, const Eigen::Matrix3d &acceleration_psd,
                    double dt, const Eigen::VectorXd &readings,
                    const std::vector<Sensor> &sensors, double alpha = 1.0,
                    double beta = 2.0, double kappa = 0.0);

  static std::map<std::string, Eigen::MatrixXd>
  sample(const AttitudeSequentialUKF &estimator, const Eigen::VectorXd &times,
         const Eigen::MatrixXd &truth, const Eigen::MatrixXd &readings,
         const std::vector<Sensor> &sensors, double interval,
         const std::set<std::string> &fields);
};

} // namespace omni::hardware::estimator

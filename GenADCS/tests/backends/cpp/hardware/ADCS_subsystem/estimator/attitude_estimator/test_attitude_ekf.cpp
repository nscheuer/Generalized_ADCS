#include "../../check.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_ekf.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace omni::hardware::estimator;

int main() {
  const Eigen::Vector4d identity(1, 0, 0, 0);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  State state{identity, zero, Matrix7::Identity()};

  const State predicted =
      AttitudeEKF::predict(state, Eigen::Matrix3d::Zero(), 0.0);
  CHECK(std::abs(predicted.quaternion.norm() - 1.0) < 1e-14);
  Eigen::Matrix<double, 7, 1> radial;
  radial << 1, 0, 0, 0, 0, 0, 0;
  CHECK((predicted.covariance * radial).norm() < 1e-12);
  CHECK(predicted.covariance.rows() == 7);

  const auto gyro = AttitudeEKF::correct_gyro(state, Eigen::Vector3d(1, 0, 0),
                                              Eigen::Matrix3d::Identity(),
                                              Eigen::Matrix3d::Identity());
  CHECK(std::abs(gyro.state.rate.x() - 0.5) < 1e-12);
  CHECK(gyro.state.covariance.rows() == 7);
  CHECK((gyro.state.covariance - gyro.state.covariance.transpose()).norm() <
        1e-13);

  Sensor sensor;
  sensor.code = 0;
  sensor.noise_cholesky = Eigen::Matrix3d::Identity();
  Eigen::VectorXd readings(3);
  readings << 1, 0, 0;
  const State stepped = AttitudeEKF::step(state, Eigen::Matrix3d::Zero(), 0.0,
                                          readings, {sensor});
  CHECK(std::abs(stepped.rate.x() - 0.5) < 1e-12);
  readings[0] = std::numeric_limits<double>::quiet_NaN();
  const State skipped = AttitudeEKF::step(state, Eigen::Matrix3d::Zero(), 0.0,
                                          readings, {sensor});
  CHECK(skipped.rate.norm() == 0.0);

  AttitudeEKF estimator(identity, zero, Matrix7::Identity(),
                        Eigen::Matrix3d::Zero());
  Eigen::VectorXd times(2);
  times << 0, 1;
  Eigen::MatrixXd truth = Eigen::MatrixXd::Zero(2, 7);
  truth.col(0).setOnes();
  Eigen::MatrixXd trajectory_readings(2, 3);
  trajectory_readings << 1, 0, 0, 0, 0, 0;
  const auto samples = AttitudeEKF::sample(
      estimator, times, truth, trajectory_readings, {sensor}, 1.0,
      {"estimated_covariance", "estimated_rate", "kalman_gain",
       "normalized_innovation_squared"});
  CHECK(samples.at("estimated_covariance").cols() == 49);
  CHECK(samples.at("kalman_gain").cols() == 18);
  CHECK(std::abs(samples.at("estimated_rate")(0, 0) - 0.5) < 1e-12);
  CHECK(std::abs(samples.at("normalized_innovation_squared")(0, 0) - 0.5) <
        1e-12);

  bool rejected = false;
  try {
    (void)AttitudeEKF::predict(state, Eigen::Matrix3d::Zero(), -1.0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}

#include "../../check.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_ekf.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace omni::hardware::estimator;

int main() {
  const Eigen::Vector4d identity(1, 0, 0, 0);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d psd = Eigen::Matrix3d::Identity() * 1e-5;

  // Constructor validation and quaternion normalization.
  const AttitudeEKF normalized(Eigen::Vector4d(2, 0, 0, 0),
                               Eigen::Vector3d(0.1, -0.2, 0.3),
                               Matrix7::Identity() * 0.1, psd);
  CHECK((normalized.quaternion - identity).norm() < 1e-14);
  const State prepared = normalized.prepare();
  CHECK((prepared.quaternion - normalized.quaternion).norm() < 1e-14);
  CHECK((prepared.rate - normalized.rate).norm() < 1e-14);
  CHECK((prepared.covariance - normalized.covariance).norm() < 1e-14);

  auto expect_invalid = [](auto fn) {
    bool rejected = false;
    try {
      (void)fn();
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected);
  };
  expect_invalid([&]() {
    return AttitudeEKF(identity, zero, Eigen::MatrixXd::Identity(6, 6), psd);
  });
  Matrix7 neg_cov = Matrix7::Identity() * 0.1;
  neg_cov(0, 0) = -1.0;
  expect_invalid([&]() { return AttitudeEKF(identity, zero, neg_cov, psd); });
  expect_invalid([&]() {
    return AttitudeEKF(identity, zero, Matrix7::Identity(), psd, {"bad_field"});
  });

  State state{identity, zero, Matrix7::Identity()};

  const State predicted =
      AttitudeEKF::predict(state, Eigen::Matrix3d::Zero(), 0.0);
  CHECK(std::abs(predicted.quaternion.norm() - 1.0) < 1e-14);
  Eigen::Matrix<double, 7, 1> radial;
  radial << 1, 0, 0, 0, 0, 0, 0;
  CHECK((predicted.covariance * radial).norm() < 1e-12);
  CHECK(predicted.covariance.rows() == 7);

  // Nonzero rate and process noise prediction.
  const State rotating{identity, Eigen::Vector3d(0.2, -0.1, 0.05),
                       Matrix7::Identity() * 0.01};
  const State propagated =
      AttitudeEKF::predict(rotating, Eigen::Matrix3d::Identity() * 2e-5, 1.0);
  CHECK(std::abs(propagated.quaternion.norm() - 1.0) < 1e-14);
  CHECK((propagated.rate - rotating.rate).norm() < 1e-14);
  CHECK((propagated.covariance.topLeftCorner<4, 4>() * propagated.quaternion)
            .norm() < 1e-12);
  CHECK((propagated.covariance - propagated.covariance.transpose()).norm() <
        1e-13);

  const auto gyro = AttitudeEKF::correct_gyro(state, Eigen::Vector3d(1, 0, 0),
                                              Eigen::Matrix3d::Identity(),
                                              Eigen::Matrix3d::Identity());
  CHECK(std::abs(gyro.state.rate.x() - 0.5) < 1e-12);
  CHECK(gyro.state.covariance.rows() == 7);
  CHECK((gyro.state.covariance - gyro.state.covariance.transpose()).norm() <
        1e-13);

  // Star tracker correction reduces attitude error and covariance.
  const Eigen::Vector4d tilted =
      Eigen::Vector4d(0.9984379, 0.04997396, -0.02498698, 0.00999479)
          .normalized();
  State tilted_state{tilted, zero, Matrix7::Identity() * 0.1};
  tilted_state.covariance.topLeftCorner<4, 4>() -=
      0.1 * tilted * tilted.transpose();
  const auto tracker = AttitudeEKF::correct_star_tracker(
      tilted_state, identity, identity, Eigen::Matrix3d::Identity() * 1e-5);
  CHECK(tracker.state.quaternion.tail<3>().norm() <
        tilted.tail<3>().norm() / 50.0);
  CHECK(tracker.state.covariance(1, 1) < tilted_state.covariance(1, 1));

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

  // Step fusing both gyro and star tracker.
  Sensor gyro_sensor;
  gyro_sensor.code = 0;
  gyro_sensor.offset = 0;
  gyro_sensor.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-3;
  Sensor tracker_sensor;
  tracker_sensor.code = 2;
  tracker_sensor.offset = 3;
  tracker_sensor.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-3;
  Eigen::VectorXd fused_readings(7);
  fused_readings << 0, 0, 0, 1, 0, 0, 0;
  const State fused = AttitudeEKF::step(
      {tilted, Eigen::Vector3d(0.5, -0.3, 0.2), Matrix7::Identity() * 0.1},
      Eigen::Matrix3d::Identity() * 1e-8, 0.1, fused_readings,
      {gyro_sensor, tracker_sensor});
  CHECK(fused.rate.norm() < 0.01);
  CHECK(fused.quaternion.tail<3>().norm() < 0.01);

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
      {"estimated_covariance", "estimated_rate", "estimated_attitude_covariance",
       "estimated_rate_covariance", "kalman_gain",
       "normalized_innovation_squared",
       "normalized_estimation_error_squared"});
  CHECK(samples.at("estimated_covariance").cols() == 49);
  CHECK(samples.at("estimated_attitude_covariance").cols() == 9);
  CHECK(samples.at("estimated_rate_covariance").cols() == 9);
  CHECK(samples.at("kalman_gain").cols() == 18);
  CHECK(std::abs(samples.at("estimated_rate")(0, 0) - 0.5) < 1e-12);
  CHECK(std::abs(samples.at("normalized_innovation_squared")(0, 0) - 0.5) <
        1e-12);
  CHECK(std::isfinite(samples.at("normalized_estimation_error_squared")(0, 0)));

  expect_invalid(
      [&]() { return AttitudeEKF::predict(state, Eigen::Matrix3d::Zero(), -1.0); });
}

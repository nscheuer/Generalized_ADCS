#include "../../check.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_mekf.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace omni::hardware::estimator;
int main() {
  State state{Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
              Matrix6::Identity()};
  auto predicted =
      AttitudeMEKF::predict(state, Eigen::Matrix3d::Identity(), 2.0);
  CHECK((predicted.quaternion - state.quaternion).norm() < 1e-14);
  CHECK(std::abs(predicted.covariance(0, 0) - (1 + 4 + 8.0 / 3)) < 1e-12);
  CHECK(std::abs(predicted.covariance(0, 3) - (-2 - 2)) < 1e-12);
  CHECK(std::abs(predicted.covariance(3, 3) - 3) < 1e-12);
  auto corrected = AttitudeMEKF::correct_gyro(state, Eigen::Vector3d(1, 0, 0),
                                              Eigen::Matrix3d::Identity(),
                                              Eigen::Matrix3d::Identity());
  CHECK(std::abs(corrected.state.rate.x() - 0.5) < 1e-12);
  CHECK(std::abs(corrected.state.covariance(3, 3) - 0.5) < 1e-12);
  CHECK(std::abs(corrected.state.quaternion[0] - 1) < 1e-12);
  Eigen::Vector4d measured(std::cos(0.1), std::sin(0.1), 0, 0);
  auto tracker = AttitudeMEKF::correct_star_tracker(
      state, measured, Eigen::Vector4d(1, 0, 0, 0),
      Eigen::Matrix3d::Identity());
  CHECK(std::abs(tracker.state.quaternion[1] - std::sin(0.05)) < 1e-12);
  Sensor sensor;
  sensor.code = 0;
  sensor.noise_cholesky = Eigen::Matrix3d::Identity();
  Eigen::VectorXd readings(3);
  readings << 1, 0, 0;
  auto stepped =
      AttitudeMEKF::step(state, Eigen::Matrix3d::Zero(), 0, readings, {sensor});
  CHECK(std::abs(stepped.rate.x() - 0.5) < 1e-12);
  readings[0] = std::numeric_limits<double>::quiet_NaN();
  stepped =
      AttitudeMEKF::step(state, Eigen::Matrix3d::Zero(), 0, readings, {sensor});
  CHECK(stepped.rate.norm() == 0);
  bool rejected = false;
  try {
    (void)AttitudeMEKF::predict(state, Eigen::Matrix3d::Zero(), -1);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  AttitudeMEKF estimator(state.quaternion, state.rate, state.covariance,
                         Eigen::Matrix3d::Zero());
  Eigen::VectorXd times(3);
  times << 0, 0.5, 1;
  Eigen::MatrixXd truth = Eigen::MatrixXd::Zero(3, 7);
  truth.col(0).setOnes();
  Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(3, 3);
  samples(0, 0) = 1;
  samples.row(2).setConstant(std::numeric_limits<double>::quiet_NaN());
  auto result = AttitudeMEKF::sample(
      estimator, times, truth, samples, {sensor}, 1,
      {"estimated_rate", "measurement_innovation_covariance",
       "normalized_innovation_squared", "measurement_residual", "kalman_gain",
       "normalized_estimation_error_squared"});
  CHECK(std::abs(result.at("estimated_rate")(0, 0) - 0.5) < 1e-12);
  CHECK(std::abs(result.at("measurement_innovation_covariance")(0, 0) - 2) <
        1e-12);
  CHECK(std::abs(result.at("normalized_innovation_squared")(0, 0) - 0.5) <
        1e-12);
  CHECK(std::abs(result.at("normalized_estimation_error_squared")(0, 0) - 0.5) <
        1e-12);
  CHECK(std::abs(result.at("measurement_residual")(0, 0) - 0.5) < 1e-12);
  CHECK(std::abs(result.at("kalman_gain")(0, 9) - 0.5) < 1e-12);
  CHECK(std::isnan(result.at("estimated_rate")(1, 0)));
  CHECK(std::isnan(result.at("measurement_residual")(2, 0)));
  CHECK(std::abs(result.at("estimated_rate")(2, 0) - 0.5) < 1e-12);
  rejected = false;
  try {
    (void)AttitudeMEKF::sample(estimator, times, truth, samples, {sensor}, 0,
                               {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  // Constructor normalization, prepare() independence, and validation
  AttitudeMEKF normalized(Eigen::Vector4d(2, 0, 0, 0),
                          Eigen::Vector3d(0.1, -0.2, 0.3),
                          Matrix6::Identity() * 0.25,
                          Eigen::Matrix3d::Identity() * 1e-4,
                          {"estimated_attitude"});
  CHECK((normalized.quaternion - Eigen::Vector4d(1, 0, 0, 0)).norm() < 1e-14);
  CHECK(normalized.record.count("estimated_attitude") == 1);
  auto prep = normalized.prepare();
  CHECK((prep.quaternion - normalized.quaternion).norm() < 1e-14);
  CHECK((prep.rate - normalized.rate).norm() < 1e-14);

  auto expect_invalid = [](auto &&fn) {
    bool threw = false;
    try {
      fn();
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    CHECK(threw);
  };
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d::Zero(), Eigen::Vector3d::Zero(),
                 Matrix6::Identity(), Eigen::Matrix3d::Identity());
  });
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d(1, 0, 0, 0),
                 Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0, 0),
                 Matrix6::Identity(), Eigen::Matrix3d::Identity());
  });
  Matrix6 asym = Matrix6::Identity();
  asym(0, 1) = 0.5;
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), asym,
                 Eigen::Matrix3d::Identity());
  });
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                 -Matrix6::Identity(), Eigen::Matrix3d::Identity());
  });
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                 Eigen::MatrixXd::Identity(7, 7), Eigen::Matrix3d::Identity());
  });
  expect_invalid([&]() {
    AttitudeMEKF(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                 Matrix6::Identity(), Eigen::Matrix3d::Identity(),
                 {"unknown_field"});
  });

  // predict with non-zero angular rate and dt = 0
  State turning_state{Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0, 0, 0.2),
                      Matrix6::Identity() * 0.2};
  auto pred_zero = AttitudeMEKF::predict(
      turning_state, Eigen::Matrix3d::Identity() * 1e-3, 0.0);
  CHECK((pred_zero.quaternion - turning_state.quaternion).norm() < 1e-14);
  CHECK((pred_zero.rate - turning_state.rate).norm() < 1e-14);
  CHECK((pred_zero.covariance - turning_state.covariance).norm() < 1e-14);
  auto pred_turn = AttitudeMEKF::predict(
      turning_state, Eigen::Matrix3d::Identity() * 1e-3, 1.0);
  CHECK(std::abs(pred_turn.quaternion[0] - std::cos(0.1)) < 1e-12);
  CHECK(std::abs(pred_turn.quaternion[3] + std::sin(0.1)) < 1e-12);

  // direct correct() and singular innovation rejection
  Eigen::Matrix<double, 3, 6> h_direct = Eigen::Matrix<double, 3, 6>::Zero();
  h_direct.leftCols<3>() = Eigen::Matrix3d::Identity();
  auto direct = AttitudeMEKF::correct(
      state, Eigen::Vector3d(0.02, -0.01, 0.03), h_direct,
      Eigen::Matrix3d::Identity() * 1e-3);
  CHECK(direct.state.quaternion.tail<3>().norm() > 0.0);
  CHECK(std::abs(direct.state.quaternion.norm() - 1.0) < 1e-14);
  CHECK((direct.state.covariance - direct.state.covariance.transpose()).norm() <
        1e-14);
  expect_invalid([&]() {
    State zero_cov{Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                   Matrix6::Zero()};
    (void)AttitudeMEKF::correct(zero_cov, Eigen::Vector3d::Zero(), h_direct,
                                Eigen::Matrix3d::Zero());
  });

  // non-identity mounting in correct_gyro and correct_star_tracker
  Eigen::Matrix3d mount_rot;
  mount_rot << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  auto gyro_mounted = AttitudeMEKF::correct_gyro(
      state, Eigen::Vector3d(0, 0.4, 0), mount_rot,
      Eigen::Matrix3d::Identity() * 1e-4);
  CHECK(std::abs(gyro_mounted.state.rate.x() - 0.4) < 1e-3);
  CHECK(std::abs(gyro_mounted.state.rate.y()) < 1e-12);

  Eigen::Vector4d mount_q(std::cos(0.2), 0, 0, std::sin(0.2));
  auto tracker_mounted = AttitudeMEKF::correct_star_tracker(
      state, mount_q, mount_q, Eigen::Matrix3d::Identity() * 1e-5);
  CHECK(tracker_mounted.state.quaternion.tail<3>().norm() < 1e-12);

  // step with disabled sensor, star tracker, mixed order, and direction sensors
  Sensor disabled_gyro;
  disabled_gyro.code = 0;
  disabled_gyro.enabled = false;
  disabled_gyro.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-3;
  State nz_rate{Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0.2, 0, 0),
                Matrix6::Identity() * 0.1};
  auto skipped = AttitudeMEKF::step(nz_rate, Eigen::Matrix3d::Zero(), 0.0,
                                    Eigen::Vector3d::Zero(), {disabled_gyro});
  CHECK((skipped.rate - nz_rate.rate).norm() < 1e-14);

  Sensor mag_sensor;
  mag_sensor.code = 1;
  mag_sensor.offset = 0;
  mag_sensor.reference = Eigen::Vector3d(1, 0, 0);
  mag_sensor.saturation = 10.0;
  mag_sensor.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-3;

  Sensor css_sensor;
  css_sensor.code = 3;
  css_sensor.offset = 3;
  css_sensor.reference = Eigen::Vector3d(0, 0, 1);
  css_sensor.normal = Eigen::Vector3d(0, 0, 1);
  css_sensor.sensitivity = 1.0;
  css_sensor.saturation = 10.0;
  css_sensor.noise_std = 1e-3;

  Sensor fss_sensor;
  fss_sensor.code = 4;
  fss_sensor.offset = 4;
  fss_sensor.reference = Eigen::Vector3d(0, 0, 1);
  fss_sensor.fov = Eigen::Vector2d(1.0, 1.0);
  fss_sensor.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-3;

  State tilted{Eigen::Vector4d(std::cos(0.05), std::sin(0.05), 0, 0),
               Eigen::Vector3d::Zero(), Matrix6::Identity() * 0.1};
  Eigen::VectorXd dir_readings(6);
  dir_readings << 1, 0, 0, 1, 0, 0;
  auto dir_stepped = AttitudeMEKF::step(
      tilted, Eigen::Matrix3d::Zero(), 0.0, dir_readings,
      {mag_sensor, css_sensor, fss_sensor});
  CHECK(dir_stepped.quaternion.tail<3>().norm() <
        tilted.quaternion.tail<3>().norm());
}

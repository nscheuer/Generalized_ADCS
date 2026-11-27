#include "../../check.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_batched_ukf.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace omni::hardware::estimator;

int main() {
  const Eigen::Vector4d identity(1, 0, 0, 0);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d psd = Eigen::Matrix3d::Identity() * 1e-4;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // 1. Constructor, COVARIANCE_SIZE, ukf_params, normalization, and validation.
  CHECK(AttitudeBatchedUKF::COVARIANCE_SIZE == 6);
  const AttitudeBatchedUKF ukf(Eigen::Vector4d(2, 0, 0, 0),
                               Eigen::Vector3d(0.1, -0.2, 0.3),
                               Matrix6::Identity() * 0.25, psd, 0.5, 2.0, 1.0,
                               {"estimated_attitude"});
  CHECK((ukf.quaternion - identity).norm() < 1e-14);
  CHECK(std::abs(ukf.alpha - 0.5) < 1e-14);
  CHECK(std::abs(ukf.beta - 2.0) < 1e-14);
  CHECK(std::abs(ukf.kappa - 1.0) < 1e-14);
  CHECK((ukf.ukf_params - Eigen::Vector3d(0.5, 2.0, 1.0)).norm() < 1e-14);
  CHECK(ukf.record.count("estimated_attitude") == 1);

  const State prepared = ukf.prepare();
  CHECK((prepared.quaternion - ukf.quaternion).norm() < 1e-14);
  CHECK((prepared.rate - ukf.rate).norm() < 1e-14);
  CHECK((prepared.covariance - ukf.covariance).norm() < 1e-14);

  auto expect_invalid = [](auto &&fn) {
    bool threw = false;
    try {
      (void)fn();
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    CHECK(threw);
  };

  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Eigen::MatrixXd::Identity(7, 7),
                              psd);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, 0.0,
                              2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, -0.5,
                              2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, nan,
                              2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, 1.0,
                              nan, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, 1.0,
                              2.0, nan);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, 1.0,
                              2.0, -6.0);
  });
  expect_invalid([&]() {
    return AttitudeBatchedUKF(identity, zero, Matrix6::Identity(), psd, 1.0,
                              2.0, 0.0, {"unknown_field"});
  });

  // 2. Prediction checks (small-angle regime matches linearized F P F^T + Q).
  const State state{identity, zero, Matrix6::Identity()};
  const State small_state{identity, zero, Matrix6::Identity() * 1e-4};
  const double dt_small = 0.1;
  const State pred_zero = AttitudeBatchedUKF::predict(
      small_state, Eigen::Matrix3d::Identity(), dt_small);
  CHECK((pred_zero.quaternion - small_state.quaternion).norm() < 1e-13);
  CHECK(std::abs(pred_zero.covariance(0, 0) -
                 (1e-4 + 1e-4 * dt_small * dt_small +
                  dt_small * dt_small * dt_small / 3.0)) < 1e-10);
  CHECK(std::abs(pred_zero.covariance(0, 3) -
                 (-1e-4 * dt_small - dt_small * dt_small / 2.0)) < 1e-10);
  CHECK(std::abs(pred_zero.covariance(3, 3) - (1e-4 + dt_small)) < 1e-10);
  CHECK((pred_zero.covariance - pred_zero.covariance.transpose()).norm() <
        1e-13);

  const State rotating{identity, Eigen::Vector3d(0.2, -0.1, 0.05),
                       Matrix6::Identity() * 0.01};
  const State pred_rot = AttitudeBatchedUKF::predict(
      rotating, Eigen::Matrix3d::Identity() * 1e-4, 0.5, 0.5, 2.0, 1.0);
  CHECK(std::abs(pred_rot.quaternion.norm() - 1.0) < 1e-14);
  CHECK((pred_rot.rate - rotating.rate).norm() < 1e-14);
  CHECK(pred_rot.quaternion[0] < 1.0);

  expect_invalid([&]() {
    return AttitudeBatchedUKF::predict(state, Eigen::Matrix3d::Zero(), -0.1);
  });

  // 3. Gyroscope (code 0) and NaN skipping.
  Sensor gyro;
  gyro.code = 0;
  gyro.noise_cholesky = Eigen::Matrix3d::Identity();
  Eigen::VectorXd gyro_readings(3);
  gyro_readings << 1.0, 0.0, 0.0;
  const State gyro_step = AttitudeBatchedUKF::step(
      state, Eigen::Matrix3d::Zero(), 0.0, gyro_readings, {gyro});
  CHECK(std::abs(gyro_step.rate.x() - 0.5) < 1e-11);
  CHECK(std::abs(gyro_step.covariance(3, 3) - 0.5) < 1e-11);

  gyro_readings[0] = nan;
  const State nan_skipped = AttitudeBatchedUKF::step(
      state, Eigen::Matrix3d::Zero(), 0.0, gyro_readings, {gyro});
  CHECK(nan_skipped.rate.norm() < 1e-14);

  // 4. Star tracker (code 2).
  Sensor tracker;
  tracker.code = 2;
  tracker.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd tracker_readings(4);
  tracker_readings << std::cos(0.1), std::sin(0.1), 0.0, 0.0;
  const State tracker_step = AttitudeBatchedUKF::step(
      {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(), 0.0,
      tracker_readings, {tracker});
  CHECK(tracker_step.quaternion[1] > 0.09);
  CHECK(tracker_step.covariance(0, 0) < 0.01);

  // 5. Magnetometer (code 1).
  Sensor mag;
  mag.code = 1;
  mag.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  mag.saturation = 10.0;
  mag.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd mag_readings(3);
  mag_readings << 0.1, 0.0, 0.995;
  const State mag_step = AttitudeBatchedUKF::step(
      {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(), 0.0,
      mag_readings, {mag});
  CHECK(mag_step.covariance(0, 0) < 0.1);
  CHECK(mag_step.covariance(1, 1) < 0.1);

  // 6. Coarse sun sensor (code 3).
  Sensor css;
  css.code = 3;
  css.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  css.normal = Eigen::Vector3d(1.0, 0.0, 0.0);
  css.sensitivity = 1.0;
  css.saturation = 5.0;
  css.noise_std = 0.05;
  Eigen::VectorXd css_readings(1);
  css_readings << 0.2;
  const State css_step = AttitudeBatchedUKF::step(
      {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(), 0.0,
      css_readings, {css});
  CHECK(css_step.covariance(1, 1) < 0.1);

  // 7. Fine sun sensor (code 4) and fine earth sensor (code 5) + FOV dropout.
  Sensor fss;
  fss.code = 4;
  fss.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  fss.fov = Eigen::Vector2d(0.8, 0.8);
  fss.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd fss_readings(2);
  fss_readings << 0.05, -0.04;
  const State fss_step = AttitudeBatchedUKF::step(
      {identity, zero, Matrix6::Identity() * 0.01}, Eigen::Matrix3d::Zero(),
      0.0, fss_readings, {fss});
  CHECK(fss_step.covariance(0, 0) < 0.01);
  CHECK(fss_step.covariance(1, 1) < 0.01);

  // FOV-boundary sigma-point dropout: large attitude covariance causes sigma
  // points to exceed FOV, so sensor update must be cleanly skipped while other
  // valid sensors in the batch still update.
  Sensor fes;
  fes.code = 5;
  fes.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  fes.fov = Eigen::Vector2d(0.1, 0.1);
  fes.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd fes_readings(2);
  fes_readings << 0.02, 0.02;
  const State large_cov{identity, zero, Matrix6::Identity() * 0.5};
  const State fes_dropped = AttitudeBatchedUKF::step(
      large_cov, Eigen::Matrix3d::Zero(), 0.0, fes_readings, {fes});
  CHECK((fes_dropped.quaternion - identity).norm() < 1e-13);
  CHECK((fes_dropped.covariance - large_cov.covariance).norm() < 1e-12);

  // 8. Order-invariance across multiple heterogeneous sensors.
  Sensor gyro_b = gyro;
  gyro_b.offset = 0;
  Sensor tracker_b = tracker;
  tracker_b.offset = 3;
  Sensor mag_b = mag;
  mag_b.offset = 7;
  Sensor css_b = css;
  css_b.offset = 10;
  Sensor fss_b = fss;
  fss_b.offset = 11;
  Sensor fes_b = fes;
  fes_b.fov = Eigen::Vector2d(0.8, 0.8);
  fes_b.offset = 13;

  Eigen::VectorXd multi_readings(15);
  multi_readings << 0.3, -0.2, 0.1, std::cos(0.05), std::sin(0.05), 0.0, 0.0,
      0.08, -0.04, 0.996, 0.15, 0.04, -0.03, 0.02, 0.01;

  const State init_multi{identity, Eigen::Vector3d(0.05, -0.05, 0.02),
                         Matrix6::Identity() * 0.02};
  const State batch_order_1 = AttitudeBatchedUKF::step(
      init_multi, psd, 0.1, multi_readings,
      {gyro_b, tracker_b, mag_b, css_b, fss_b, fes_b});
  const State batch_order_2 = AttitudeBatchedUKF::step(
      init_multi, psd, 0.1, multi_readings,
      {fes_b, css_b, tracker_b, gyro_b, fss_b, mag_b});
  CHECK((batch_order_1.quaternion - batch_order_2.quaternion).norm() < 1e-13);
  CHECK((batch_order_1.rate - batch_order_2.rate).norm() < 1e-13);
  CHECK((batch_order_1.covariance - batch_order_2.covariance).norm() < 1e-13);

  // Out-of-range reading offset throws.
  expect_invalid([&]() {
    Sensor bad = gyro;
    bad.offset = 1;
    return AttitudeBatchedUKF::step(state, Eigen::Matrix3d::Zero(), 0.0,
                                    gyro_readings, {bad});
  });

  // 9. Trajectory sampling & all diagnostic fields.
  const AttitudeBatchedUKF estimator(identity, zero, Matrix6::Identity(),
                                     Eigen::Matrix3d::Zero(), 0.8, 2.0, 0.0);
  Eigen::VectorXd times(3);
  times << 0.0, 0.5, 1.0;
  Eigen::MatrixXd truth = Eigen::MatrixXd::Zero(3, 7);
  truth.col(0).setOnes();
  Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(3, 3);
  samples(0, 0) = 1.0;
  samples.row(2).setConstant(nan);

  const auto diag = AttitudeBatchedUKF::sample(
      estimator, times, truth, samples, {gyro}, 1.0,
      {"estimated_attitude", "estimated_rate", "estimated_covariance",
       "estimated_attitude_covariance", "estimated_rate_covariance",
       "measurement_innovation", "measurement_innovation_covariance",
       "normalized_innovation_squared", "normalized_estimation_error_squared",
       "measurement_residual", "kalman_gain"});
  CHECK(diag.at("estimated_attitude").cols() == 4);
  CHECK(diag.at("estimated_covariance").cols() == 36);
  CHECK(std::abs(diag.at("estimated_rate")(0, 0) - 0.5) < 1e-11);
  CHECK(std::abs(diag.at("measurement_innovation_covariance")(0, 0) - 2.0) <
        1e-11);
  CHECK(std::abs(diag.at("normalized_innovation_squared")(0, 0) - 0.5) < 1e-11);
  CHECK(std::abs(diag.at("normalized_estimation_error_squared")(0, 0) - 0.5) <
        1e-11);
  CHECK(std::abs(diag.at("measurement_residual")(0, 0) - 0.5) < 1e-11);
  CHECK(std::abs(diag.at("kalman_gain")(0, 9) - 0.5) < 1e-11);
  CHECK(std::isnan(diag.at("estimated_rate")(1, 0)));
  CHECK(std::isnan(diag.at("measurement_residual")(2, 0)));
  CHECK(std::abs(diag.at("estimated_rate")(2, 0) - 0.5) < 1e-11);

  return 0;
}

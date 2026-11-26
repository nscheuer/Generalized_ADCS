#include "../../check.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_sequential_srukf.hpp"
#include "hardware/ADCS_subsystem/estimator/attitude_estimator/attitude_sequential_ukf.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace omni::hardware::estimator;

namespace {

void check_state_parity(const State &a, const State &b, double q_tol = 1e-11,
                        double w_tol = 1e-11, double p_tol = 1e-10) {
  const double q_sign = a.quaternion.dot(b.quaternion) >= 0.0 ? 1.0 : -1.0;
  CHECK((a.quaternion - q_sign * b.quaternion).norm() < q_tol);
  CHECK((a.rate - b.rate).norm() < w_tol);
  CHECK((a.covariance - b.covariance).norm() < p_tol);
  CHECK((a.covariance - a.covariance.transpose()).norm() < 1e-13);
}

} // namespace

int main() {
  const Eigen::Vector4d identity(1, 0, 0, 0);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d psd = Eigen::Matrix3d::Identity() * 1e-4;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // 1. Constructor, COVARIANCE_SIZE, ukf_params, normalization, and validation.
  CHECK(AttitudeSequentialSRUKF::COVARIANCE_SIZE == 6);
  const AttitudeSequentialSRUKF srukf(Eigen::Vector4d(2, 0, 0, 0),
                                      Eigen::Vector3d(0.1, -0.2, 0.3),
                                      Matrix6::Identity() * 0.25, psd, 0.5, 2.0,
                                      1.0, {"estimated_attitude"});
  CHECK((srukf.quaternion - identity).norm() < 1e-14);
  CHECK(std::abs(srukf.alpha - 0.5) < 1e-14);
  CHECK(std::abs(srukf.beta - 2.0) < 1e-14);
  CHECK(std::abs(srukf.kappa - 1.0) < 1e-14);
  CHECK((srukf.ukf_params - Eigen::Vector3d(0.5, 2.0, 1.0)).norm() < 1e-14);
  CHECK(srukf.record.count("estimated_attitude") == 1);

  const State prepared = srukf.prepare();
  CHECK((prepared.quaternion - srukf.quaternion).norm() < 1e-14);
  CHECK((prepared.rate - srukf.rate).norm() < 1e-14);
  CHECK((prepared.covariance - srukf.covariance).norm() < 1e-14);

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
    return AttitudeSequentialSRUKF(identity, zero,
                                   Eigen::MatrixXd::Identity(7, 7), psd);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   0.0, 2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   -0.5, 2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   nan, 2.0, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   1.0, nan, 0.0);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   1.0, 2.0, nan);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   1.0, 2.0, -6.0);
  });
  expect_invalid([&]() {
    return AttitudeSequentialSRUKF(identity, zero, Matrix6::Identity(), psd,
                                   1.0, 2.0, 0.0, {"unknown_field"});
  });

  // 2. Prediction checks and parity with UKF for alpha=1.0 and alpha=0.5.
  const State state{identity, zero, Matrix6::Identity()};
  const State small_state{identity, zero, Matrix6::Identity() * 1e-4};
  const double dt_small = 0.1;
  const State pred_zero = AttitudeSequentialSRUKF::predict(
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
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_pred = AttitudeSequentialSRUKF::predict(
        rotating, Eigen::Matrix3d::Identity() * 1e-4, 0.5, alpha_val, 2.0, 1.0);
    const State ukf_pred = AttitudeSequentialUKF::predict(
        rotating, Eigen::Matrix3d::Identity() * 1e-4, 0.5, alpha_val, 2.0, 1.0);
    check_state_parity(srukf_pred, ukf_pred);
  }

  expect_invalid([&]() {
    return AttitudeSequentialSRUKF::predict(state, Eigen::Matrix3d::Zero(),
                                            -0.1);
  });

  // 3. Gyroscope (code 0) and NaN skipping.
  Sensor gyro;
  gyro.code = 0;
  gyro.noise_cholesky = Eigen::Matrix3d::Identity();
  Eigen::VectorXd gyro_readings(3);
  gyro_readings << 1.0, 0.0, 0.0;
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_gyro = AttitudeSequentialSRUKF::step(
        state, Eigen::Matrix3d::Zero(), 0.0, gyro_readings, {gyro}, alpha_val);
    const State ukf_gyro = AttitudeSequentialUKF::step(
        state, Eigen::Matrix3d::Zero(), 0.0, gyro_readings, {gyro}, alpha_val);
    check_state_parity(srukf_gyro, ukf_gyro);
    CHECK(std::abs(srukf_gyro.rate.x() - 0.5) < 1e-11);
    CHECK(std::abs(srukf_gyro.covariance(3, 3) - 0.5) < 1e-11);
  }

  gyro_readings[0] = nan;
  const State nan_skipped = AttitudeSequentialSRUKF::step(
      state, Eigen::Matrix3d::Zero(), 0.0, gyro_readings, {gyro});
  CHECK(nan_skipped.rate.norm() < 1e-14);

  // 4. Star tracker (code 2).
  Sensor tracker;
  tracker.code = 2;
  tracker.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd tracker_readings(4);
  tracker_readings << std::cos(0.1), std::sin(0.1), 0.0, 0.0;
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_tracker = AttitudeSequentialSRUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, tracker_readings, {tracker}, alpha_val);
    const State ukf_tracker = AttitudeSequentialUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, tracker_readings, {tracker}, alpha_val);
    check_state_parity(srukf_tracker, ukf_tracker);
    CHECK(srukf_tracker.quaternion[1] > 0.09);
    CHECK(srukf_tracker.covariance(0, 0) < 0.01);
  }

  // 5. Magnetometer (code 1).
  Sensor mag;
  mag.code = 1;
  mag.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  mag.saturation = 10.0;
  mag.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd mag_readings(3);
  mag_readings << 0.1, 0.0, 0.995;
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_mag = AttitudeSequentialSRUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, mag_readings, {mag}, alpha_val);
    const State ukf_mag = AttitudeSequentialUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, mag_readings, {mag}, alpha_val);
    check_state_parity(srukf_mag, ukf_mag);
    CHECK(srukf_mag.covariance(0, 0) < 0.1);
    CHECK(srukf_mag.covariance(1, 1) < 0.1);
  }

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
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_css = AttitudeSequentialSRUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, css_readings, {css}, alpha_val);
    const State ukf_css = AttitudeSequentialUKF::step(
        {identity, zero, Matrix6::Identity() * 0.1}, Eigen::Matrix3d::Zero(),
        0.0, css_readings, {css}, alpha_val);
    check_state_parity(srukf_css, ukf_css);
    CHECK(srukf_css.covariance(1, 1) < 0.1);
  }

  // 7. Fine sun sensor (code 4) and fine earth sensor (code 5) + FOV dropout.
  Sensor fss;
  fss.code = 4;
  fss.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  fss.fov = Eigen::Vector2d(0.8, 0.8);
  fss.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd fss_readings(2);
  fss_readings << 0.05, -0.04;
  for (double alpha_val : {1.0, 0.5}) {
    const State srukf_fss = AttitudeSequentialSRUKF::step(
        {identity, zero, Matrix6::Identity() * 0.01}, Eigen::Matrix3d::Zero(),
        0.0, fss_readings, {fss}, alpha_val);
    const State ukf_fss = AttitudeSequentialUKF::step(
        {identity, zero, Matrix6::Identity() * 0.01}, Eigen::Matrix3d::Zero(),
        0.0, fss_readings, {fss}, alpha_val);
    check_state_parity(srukf_fss, ukf_fss);
    CHECK(srukf_fss.covariance(0, 0) < 0.01);
    CHECK(srukf_fss.covariance(1, 1) < 0.01);
  }

  // FOV-boundary sigma-point dropout: large attitude covariance causes sigma
  // points to exceed FOV, so sensor update must be cleanly skipped.
  Sensor fes;
  fes.code = 5;
  fes.reference = Eigen::Vector3d(0.0, 0.0, 1.0);
  fes.fov = Eigen::Vector2d(0.1, 0.1);
  fes.noise_cholesky = Eigen::Matrix3d::Identity() * 1e-2;
  Eigen::VectorXd fes_readings(2);
  fes_readings << 0.02, 0.02;
  const State large_cov{identity, zero, Matrix6::Identity() * 0.5};
  const State fes_dropped = AttitudeSequentialSRUKF::step(
      large_cov, Eigen::Matrix3d::Zero(), 0.0, fes_readings, {fes});
  CHECK((fes_dropped.quaternion - identity).norm() < 1e-13);
  CHECK((fes_dropped.covariance - large_cov.covariance).norm() < 1e-12);

  // Out-of-range reading offset throws.
  expect_invalid([&]() {
    Sensor bad = gyro;
    bad.offset = 1;
    return AttitudeSequentialSRUKF::step(state, Eigen::Matrix3d::Zero(), 0.0,
                                         gyro_readings, {bad});
  });

  // 8. Trajectory sampling & all diagnostic fields, plus parity with UKF.
  for (double alpha_val : {1.0, 0.5}) {
    const AttitudeSequentialSRUKF estimator(
        identity, zero, Matrix6::Identity(), Eigen::Matrix3d::Zero(),
        alpha_val, 2.0, 0.0);
    const AttitudeSequentialUKF ukf_estimator(
        identity, zero, Matrix6::Identity(), Eigen::Matrix3d::Zero(),
        alpha_val, 2.0, 0.0);
    Eigen::VectorXd times(3);
    times << 0.0, 0.5, 1.0;
    Eigen::MatrixXd truth = Eigen::MatrixXd::Zero(3, 7);
    truth.col(0).setOnes();
    Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(3, 3);
    samples(0, 0) = 1.0;
    samples.row(2).setConstant(nan);

    const std::set<std::string> fields{
        "estimated_attitude", "estimated_rate", "estimated_covariance",
        "estimated_attitude_covariance", "estimated_rate_covariance",
        "measurement_innovation", "measurement_innovation_covariance",
        "normalized_innovation_squared", "normalized_estimation_error_squared",
        "measurement_residual", "kalman_gain"};
    const auto diag = AttitudeSequentialSRUKF::sample(
        estimator, times, truth, samples, {gyro}, 1.0, fields);
    const auto ukf_diag = AttitudeSequentialUKF::sample(
        ukf_estimator, times, truth, samples, {gyro}, 1.0, fields);
    CHECK(diag.at("estimated_attitude").cols() == 4);
    CHECK(diag.at("estimated_covariance").cols() == 36);
    CHECK(std::abs(diag.at("estimated_rate")(0, 0) - 0.5) < 1e-11);
    CHECK(std::abs(diag.at("measurement_innovation_covariance")(0, 0) - 2.0) <
          1e-11);
    CHECK(std::abs(diag.at("normalized_innovation_squared")(0, 0) - 0.5) <
          1e-11);
    CHECK(std::abs(diag.at("normalized_estimation_error_squared")(0, 0) - 0.5) <
          1e-11);
    CHECK(std::abs(diag.at("measurement_residual")(0, 0) - 0.5) < 1e-11);
    CHECK(std::abs(diag.at("kalman_gain")(0, 9) - 0.5) < 1e-11);
    CHECK(std::isnan(diag.at("estimated_rate")(1, 0)));
    CHECK(std::isnan(diag.at("measurement_residual")(2, 0)));
    CHECK(std::abs(diag.at("estimated_rate")(2, 0) - 0.5) < 1e-11);
    CHECK((diag.at("estimated_covariance").row(0) -
           ukf_diag.at("estimated_covariance").row(0))
              .norm() < 1e-10);
    CHECK((diag.at("estimated_covariance").row(2) -
           ukf_diag.at("estimated_covariance").row(2))
              .norm() < 1e-10);
  }

  return 0;
}

#pragma once
#include "attitude_mekf.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace omni::hardware::estimator {
struct DirectionCorrection {
  State state;
  Eigen::VectorXd innovation;
  Eigen::MatrixXd innovation_covariance;
  Eigen::MatrixXd gain;
  Eigen::VectorXd residual;
};
inline Eigen::VectorXd direction_prediction(const Sensor &sensor,
                                            const Eigen::Vector4d &q) {
  const Eigen::Quaterniond attitude(q[0], q[1], q[2], q[3]);
  const auto mount =
      (Eigen::Quaterniond(sensor.misalignment[0], sensor.misalignment[1],
                          sensor.misalignment[2], sensor.misalignment[3]) *
       Eigen::Quaterniond(sensor.orientation[0], sensor.orientation[1],
                          sensor.orientation[2], sensor.orientation[3]))
          .normalized();
  const Eigen::Vector3d direction = mount.toRotationMatrix() *
                                    attitude.normalized().toRotationMatrix() *
                                    sensor.reference;
  if (sensor.code == 1) {
    return (direction + sensor.bias)
        .cwiseMax(-sensor.saturation)
        .cwiseMin(sensor.saturation);
  }
  if (sensor.code == 3) {
    Eigen::VectorXd result(1);
    result[0] = std::clamp(sensor.sensitivity *
                                   std::max(0.0, sensor.normal.dot(direction)) +
                               sensor.bias[0],
                           0.0, sensor.saturation);
    return result;
  }
  const double angle_x = std::atan2(direction[0], direction[2]);
  const double angle_y = std::atan2(direction[1], direction[2]);
  // Match fine_direction_available(): unavailable directions produce no
  // predicted measurement, so the caller skips this correction.
  if (std::abs(angle_x) > sensor.fov[0] ||
      std::abs(angle_y) > sensor.fov[1]) {
    return Eigen::Vector2d::Constant(
        std::numeric_limits<double>::quiet_NaN());
  }
  Eigen::Vector2d result;
  result << angle_x + sensor.bias[0], angle_y + sensor.bias[1];
  return result;
}
inline DirectionCorrection correct_direction(const State &prior,
                                             const Sensor &sensor,
                                             const Eigen::VectorXd &observed) {
  const bool full = prior.covariance.rows() == 7;
  const int n = full ? 7 : 6, width = observed.size();
  const double epsilon = 1e-7;
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(width, n);
  for (int j = 0; j < (full ? 4 : 3); ++j) {
    Eigen::Vector4d plus = prior.quaternion, minus = prior.quaternion;
    if (full) {
      plus[j] += epsilon;
      minus[j] -= epsilon;
      plus.normalize();
      minus.normalize();
    } else {
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      d[j] = epsilon;
      const Eigen::Quaterniond qp(Eigen::AngleAxisd(d.norm(), d.normalized()));
      const Eigen::Quaterniond qm(Eigen::AngleAxisd(d.norm(), -d.normalized()));
      auto q = Eigen::Quaterniond(prior.quaternion[0], prior.quaternion[1],
                                  prior.quaternion[2], prior.quaternion[3]);
      auto p = (qp * q).normalized(), m = (qm * q).normalized();
      plus = {p.w(), p.x(), p.y(), p.z()};
      minus = {m.w(), m.x(), m.y(), m.z()};
    }
    h.col(j) = (direction_prediction(sensor, plus) -
                direction_prediction(sensor, minus)) /
               (2 * epsilon);
  }
  const Eigen::VectorXd expected =
      direction_prediction(sensor, prior.quaternion);
  Eigen::VectorXd residual = observed - expected;
  if (sensor.code == 4 || sensor.code == 5)
    for (int i = 0; i < width; ++i)
      residual[i] = std::remainder(residual[i], 2 * 3.14159265358979323846);
  Eigen::MatrixXd r =
      sensor.noise_cholesky.topLeftCorner(width, width) *
      sensor.noise_cholesky.topLeftCorner(width, width).transpose();
  if (sensor.code == 3)
    r(0, 0) = sensor.noise_std * sensor.noise_std;
  const Eigen::MatrixXd s = h * prior.covariance * h.transpose() + r;
  const auto solver = s.ldlt();
  if (solver.info() != Eigen::Success ||
      (solver.vectorD().array().abs() < 1e-20).any())
    throw std::invalid_argument("direction innovation covariance is singular");
  const Eigen::MatrixXd gain =
      solver.solve((h * prior.covariance).eval()).transpose();
  const Eigen::VectorXd delta = gain * residual;
  State next = prior;
  Eigen::MatrixXd a = Eigen::MatrixXd::Identity(n, n) - gain * h;
  Eigen::MatrixXd p =
      a * prior.covariance * a.transpose() + gain * r * gain.transpose();
  if (full) {
    next.quaternion = (prior.quaternion + delta.head(4)).normalized();
    next.rate += delta.tail(3);
    Eigen::MatrixXd reset = Eigen::MatrixXd::Identity(7, 7);
    reset.topLeftCorner(4, 4) -= next.quaternion * next.quaternion.transpose();
    p = reset * p * reset.transpose();
  } else {
    const Eigen::Vector3d d = delta.head(3);
    const Eigen::Quaterniond dq =
        d.norm() < 1e-12
            ? Eigen::Quaterniond(1., d[0] / 2, d[1] / 2, d[2] / 2).normalized()
            : Eigen::Quaterniond(Eigen::AngleAxisd(d.norm(), d.normalized()));
    const auto q =
        dq * Eigen::Quaterniond(prior.quaternion[0], prior.quaternion[1],
                                prior.quaternion[2], prior.quaternion[3]);
    next.quaternion = {q.w(), q.x(), q.y(), q.z()};
    next.rate += delta.tail(3);
    Eigen::MatrixXd reset = Eigen::MatrixXd::Identity(6, 6);
    Eigen::Matrix3d skew;
    skew << 0, -d[2], d[1], d[2], 0, -d[0], -d[1], d[0], 0;
    reset.topLeftCorner(3, 3) += 0.5 * skew;
    p = reset * p * reset.transpose();
  }
  next.covariance = 0.5 * (p + p.transpose()).eval();
  return {next, residual, s, gain, residual - h * delta};
}
} // namespace omni::hardware::estimator

#include "attitude_mekf.hpp"
#include "direction_correction.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omni::hardware::estimator {
AttitudeMEKF::AttitudeMEKF(Eigen::Vector4d q, Eigen::Vector3d w,
                           Eigen::MatrixXd p, Eigen::Matrix3d a,
                           std::set<std::string> record)
    : AttitudeEstimator(std::move(q), std::move(w), std::move(p), std::move(a),
                        std::move(record)) {
  if (covariance.rows() != 6) {
    throw std::invalid_argument("MEKF covariance must be 6x6");
  }
}

namespace {
using Q = Eigen::Quaterniond;
Q quat(const Eigen::Vector4d &v) { return Q(v[0], v[1], v[2], v[3]); }
Eigen::Vector4d vec(Q q) {
  q.normalize();
  return {q.w(), q.x(), q.y(), q.z()};
}
Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d s;
  s << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return s;
}
Q expmap(const Eigen::Vector3d &v) {
  double angle = v.norm();
  if (angle < 1e-12) {
    return Q(1, 0.5 * v.x(), 0.5 * v.y(), 0.5 * v.z()).normalized();
  }
  return Q(Eigen::AngleAxisd(angle, v / angle));
}
Eigen::Vector3d logmap(Q q) {
  q.normalize();
  if (q.w() < 0) {
    q.coeffs() *= -1;
  }
  const double n = q.vec().norm();
  if (n < 1e-12) {
    return 2 * q.vec();
  }
  return 2 * std::atan2(n, q.w()) * q.vec() / n;
}
Eigen::Matrix3d left_jacobian(const Eigen::Vector3d &v) {
  const double angle = v.norm();
  const Eigen::Matrix3d k = skew(v);
  if (angle < 1e-6) {
    return Eigen::Matrix3d::Identity() + 0.5 * k + k * k / 6.0;
  }
  return Eigen::Matrix3d::Identity() +
         ((1 - std::cos(angle)) / (angle * angle)) * k +
         ((angle - std::sin(angle)) / (angle * angle * angle)) * k * k;
}
} // namespace
State AttitudeMEKF::predict(const State &state, const Eigen::Matrix3d &a,
                            double dt) {
  if (!std::isfinite(dt) || dt < 0) {
    throw std::invalid_argument("dt must be finite and nonnegative");
  }
  const Eigen::Vector3d rotation = -state.rate * dt;
  const Q increment = expmap(rotation);
  Matrix6 phi = Matrix6::Identity(), noise = Matrix6::Zero();
  phi.topLeftCorner<3, 3>() = increment.toRotationMatrix();
  phi.topRightCorner<3, 3>() = -left_jacobian(rotation) * dt;
  noise.topLeftCorner<3, 3>() = a * (dt * dt * dt / 3.0);
  noise.topRightCorner<3, 3>() = -a * (dt * dt / 2.0);
  noise.bottomLeftCorner<3, 3>() = noise.topRightCorner<3, 3>();
  noise.bottomRightCorner<3, 3>() = a * dt;
  Matrix6 p = phi * state.covariance * phi.transpose() + noise;
  return {vec(increment * quat(state.quaternion)), state.rate,
          0.5 * (p + p.transpose())};
}
Correction AttitudeMEKF::correct(const State &state,
                                 const Eigen::Vector3d &residual,
                                 const Eigen::Matrix<double, 3, 6> &h,
                                 const Eigen::Matrix3d &r) {
  const Eigen::Matrix3d s = h * state.covariance * h.transpose() + r;
  const auto solver = s.ldlt();
  if (solver.info() != Eigen::Success ||
      (solver.vectorD().array().abs() < 1e-15).any()) {
    throw std::invalid_argument("innovation covariance is singular");
  }
  const Eigen::Matrix<double, 6, 3> gain =
      solver.solve((h * state.covariance).eval()).transpose();
  const Vector6 delta = gain * residual;
  const Matrix6 remainder = Matrix6::Identity() - gain * h;
  Matrix6 p = remainder * state.covariance * remainder.transpose() +
              gain * r * gain.transpose();
  Matrix6 reset = Matrix6::Identity();
  reset.topLeftCorner<3, 3>() += 0.5 * skew(delta.head<3>());
  p = reset * p * reset.transpose();
  State result{vec(expmap(delta.head<3>()) * quat(state.quaternion)),
               state.rate + delta.tail<3>(), 0.5 * (p + p.transpose())};
  return {result, residual, s, gain, residual - h * delta};
}
Correction AttitudeMEKF::correct_gyro(const State &state,
                                      const Eigen::Vector3d &measured_rate,
                                      const Eigen::Matrix3d &mounting,
                                      const Eigen::Matrix3d &covariance) {
  Eigen::Matrix<double, 3, 6> h = Eigen::Matrix<double, 3, 6>::Zero();
  h.rightCols<3>() = mounting;
  return correct(state, measured_rate - mounting * state.rate, h, covariance);
}
Correction
AttitudeMEKF::correct_star_tracker(const State &state,
                                   const Eigen::Vector4d &measured_quaternion,
                                   const Eigen::Vector4d &mounting_quaternion,
                                   const Eigen::Matrix3d &covariance) {
  const Q mount = quat(mounting_quaternion).normalized();
  const Q predicted = mount * quat(state.quaternion);
  Eigen::Matrix<double, 3, 6> h = Eigen::Matrix<double, 3, 6>::Zero();
  h.leftCols<3>() = mount.toRotationMatrix();
  return correct(state,
                 logmap(quat(measured_quaternion) * predicted.conjugate()), h,
                 covariance);
}
State AttitudeMEKF::step(const State &state, const Eigen::Matrix3d &a,
                         double dt, const Eigen::VectorXd &readings,
                         const std::vector<Sensor> &sensors) {
  State current = predict(state, a, dt);
  for (const Sensor &sensor : sensors) {
    if (!sensor.enabled || sensor.code < 0 || sensor.code > 5) {
      continue;
    }
    const Eigen::Index width = sensor.code == 0 || sensor.code == 1 ? 3
                               : sensor.code == 2                   ? 4
                               : sensor.code == 3                   ? 1
                                                                    : 2;
    if (sensor.offset < 0 || sensor.offset + width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const Eigen::VectorXd value = readings.segment(sensor.offset, width);
    if (!value.allFinite()) {
      continue;
    }
    if (sensor.code == 1 || sensor.code == 3 || sensor.code == 4 ||
        sensor.code == 5) {
      if (!direction_prediction(sensor, current.quaternion).allFinite())
        continue;
      current = correct_direction(current, sensor, value).state;
      continue;
    }
    const Q mount =
        (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
    const Eigen::Matrix3d noise =
        sensor.noise_cholesky * sensor.noise_cholesky.transpose();
    if (sensor.code == 0) {
      current = correct_gyro(current, value.head<3>() - sensor.bias,
                             mount.toRotationMatrix(), noise)
                    .state;
    } else {
      current =
          correct_star_tracker(current, value.head<4>(), vec(mount), noise)
              .state;
    }
  }
  return current;
}
} // namespace omni::hardware::estimator

namespace omni::hardware::estimator {
std::map<std::string, Eigen::MatrixXd>
AttitudeMEKF::sample(const AttitudeMEKF &estimator,
                     const Eigen::VectorXd &times, const Eigen::MatrixXd &truth,
                     const Eigen::MatrixXd &readings,
                     const std::vector<Sensor> &sensors, double interval,
                     const std::set<std::string> &fields) {
  if (!std::isfinite(interval) || interval <= 0 ||
      truth.rows() != times.size() || truth.cols() < 7 ||
      readings.rows() != times.size()) {
    throw std::invalid_argument(
        "invalid estimator sample trajectory or interval");
  }
  auto output = allocate_samples(times.size(), fields);
  State current = estimator.prepare();
  double previous = 0;
  for (Eigen::Index row = 0; row < times.size(); ++row) {
    const double time = times[row];
    if (!std::isfinite(time) || time < 0 || (row && time < times[row - 1])) {
      throw std::invalid_argument(
          "estimator times must be finite, nonnegative and ordered");
    }
    if (std::abs(time / interval - std::round(time / interval)) > 1e-9) {
      continue;
    }
    update_sample(current, estimator.acceleration_psd, time - previous,
                  truth.row(row).transpose(), readings.row(row).transpose(),
                  sensors, output, row);
    previous = time;
  }
  return output;
}
std::map<std::string, Eigen::MatrixXd>
AttitudeMEKF::allocate_samples(Eigen::Index rows,
                               const std::set<std::string> &fields) {
  const std::map<std::string, int> widths{
      {"estimated_attitude", 4},
      {"estimated_rate", 3},
      {"estimated_covariance", 36},
      {"estimated_attitude_covariance", 9},
      {"estimated_rate_covariance", 9},
      {"measurement_innovation", 3},
      {"measurement_innovation_covariance", 9},
      {"normalized_innovation_squared", 1},
      {"normalized_estimation_error_squared", 1},
      {"measurement_residual", 3},
      {"kalman_gain", 18}};
  std::map<std::string, Eigen::MatrixXd> output;
  for (const auto &name : fields) {
    if (!widths.count(name)) {
      throw std::invalid_argument("unknown estimator diagnostic field: " +
                                  name);
    }
    output[name] = Eigen::MatrixXd::Constant(
        rows, widths.at(name), std::numeric_limits<double>::quiet_NaN());
  }
  return output;
}
void AttitudeMEKF::update_sample(State &current,
                                 const Eigen::Matrix3d &acceleration_psd,
                                 double dt, const Eigen::VectorXd &truth,
                                 const Eigen::VectorXd &readings,
                                 const std::vector<Sensor> &sensors,
                                 std::map<std::string, Eigen::MatrixXd> &output,
                                 Eigen::Index row) {
  current = predict(current, acceleration_psd, dt);
  auto store = [&](const std::string &name, const auto &value) {
    auto found = output.find(name);
    if (found == output.end()) {
      return;
    }
    Eigen::Index index = 0;
    for (Eigen::Index i = 0; i < value.rows(); ++i) {
      for (Eigen::Index j = 0; j < value.cols(); ++j) {
        found->second(row, index++) = value(i, j);
      }
    }
  };
  auto scalar = [&](const std::string &name, double value) {
    auto found = output.find(name);
    if (found != output.end()) {
      found->second(row, 0) = value;
    }
  };
  for (const auto &sensor : sensors) {
    if (!sensor.enabled || sensor.code > 5 || sensor.code < 0) {
      continue;
    }
    const int width = sensor.code == 0 || sensor.code == 1 ? 3
                      : sensor.code == 2                   ? 4
                      : sensor.code == 3                   ? 1
                                                           : 2;
    if (sensor.offset < 0 || sensor.offset + width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const Eigen::VectorXd value = readings.segment(sensor.offset, width);
    if (!value.allFinite()) {
      continue;
    }
    if (sensor.code == 1 || sensor.code == 3 || sensor.code == 4 ||
        sensor.code == 5) {
      if (!direction_prediction(sensor, current.quaternion).allFinite())
        continue;
      const DirectionCorrection correction =
          correct_direction(current, sensor, value);
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      Eigen::Vector3d innovation = Eigen::Vector3d::Constant(nan);
      Eigen::Vector3d residual = Eigen::Vector3d::Constant(nan);
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Constant(nan);
      Eigen::Matrix<double, 6, 3> gain =
          Eigen::Matrix<double, 6, 3>::Constant(nan);
      const int m = width;
      innovation.head(m) = correction.innovation;
      residual.head(m) = correction.residual;
      covariance.topLeftCorner(m, m) = correction.innovation_covariance;
      gain.leftCols(m) = correction.gain;
      store("measurement_innovation", innovation);
      store("measurement_innovation_covariance", covariance);
      store("measurement_residual", residual);
      store("kalman_gain", gain);
      if (output.count("normalized_innovation_squared"))
        scalar("normalized_innovation_squared",
               correction.innovation.dot(
                   correction.innovation_covariance.ldlt().solve(
                       correction.innovation)));
      current = correction.state;
      continue;
    }
    const Q mount =
        (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
    const Eigen::Matrix3d noise =
        sensor.noise_cholesky * sensor.noise_cholesky.transpose();
    const Correction correction =
        sensor.code == 0
            ? correct_gyro(current, value.head<3>() - sensor.bias,
                           mount.toRotationMatrix(), noise)
            : correct_star_tracker(current, value.head<4>(), vec(mount), noise);
    store("measurement_innovation", correction.innovation);
    store("measurement_innovation_covariance",
          correction.innovation_covariance);
    store("measurement_residual", correction.residual);
    store("kalman_gain", correction.gain);
    if (output.count("normalized_innovation_squared")) {
      scalar("normalized_innovation_squared",
             correction.innovation.dot(
                 correction.innovation_covariance.ldlt().solve(
                     correction.innovation)));
    }
    current = correction.state;
  }
  store("estimated_attitude", current.quaternion);
  store("estimated_rate", current.rate);
  store("estimated_covariance", current.covariance);
  store("estimated_attitude_covariance",
        current.covariance.topLeftCorner<3, 3>());
  store("estimated_rate_covariance",
        current.covariance.bottomRightCorner<3, 3>());
  if (output.count("normalized_estimation_error_squared")) {
    Vector6 error;
    error.head<3>() =
        logmap(quat(truth.head<4>()) * quat(current.quaternion).conjugate());
    error.tail<3>() = truth.segment<3>(4) - current.rate;
    const auto solver = current.covariance.ldlt();
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array() == 0).any()) {
      throw std::invalid_argument("estimator covariance is singular");
    }
    scalar("normalized_estimation_error_squared",
           error.dot(solver.solve(error)));
  }
}
} // namespace omni::hardware::estimator

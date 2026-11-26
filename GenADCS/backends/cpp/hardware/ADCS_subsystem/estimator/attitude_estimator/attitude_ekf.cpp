#include "attitude_ekf.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omni::hardware::estimator {
namespace {
using Q = Eigen::Quaterniond;
using Matrix3x7 = Eigen::Matrix<double, 3, 7>;
using Matrix7x3 = Eigen::Matrix<double, 7, 3>;
using Matrix6x7 = Eigen::Matrix<double, 6, 7>;
using Matrix7x6 = Eigen::Matrix<double, 7, 6>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector7d = Eigen::Matrix<double, 7, 1>;

Q quat(const Eigen::Vector4d &value) {
  return Q(value[0], value[1], value[2], value[3]);
}
Eigen::Vector4d vec(Q value) {
  value.normalize();
  return {value.w(), value.x(), value.y(), value.z()};
}
Eigen::Matrix3d skew(const Eigen::Vector3d &value) {
  Eigen::Matrix3d result;
  result << 0.0, -value.z(), value.y(), value.z(), 0.0, -value.x(), -value.y(),
      value.x(), 0.0;
  return result;
}
Q expmap(const Eigen::Vector3d &value) {
  const double angle = value.norm();
  if (angle < 1e-12) {
    return Q(1.0, 0.5 * value.x(), 0.5 * value.y(), 0.5 * value.z())
        .normalized();
  }
  return Q(Eigen::AngleAxisd(angle, value / angle));
}
Eigen::Vector3d logmap(Q value) {
  value.normalize();
  if (value.w() < 0.0) {
    value.coeffs() *= -1.0;
  }
  const double vector_norm = value.vec().norm();
  if (vector_norm < 1e-12) {
    return 2.0 * value.vec();
  }
  return 2.0 * std::atan2(vector_norm, value.w()) * value.vec() / vector_norm;
}
Eigen::Vector4d propagate_quaternion(const Eigen::Vector4d &q,
                                     const Eigen::Vector3d &w, double dt) {
  return vec(expmap(-w * dt) * quat(q));
}
Vector7d transition(const Vector7d &x, double dt) {
  Vector7d result;
  result.head<4>() = propagate_quaternion(x.head<4>(), x.tail<3>(), dt);
  if (result.head<4>().dot(x.head<4>()) < 0.0) {
    result.head<4>() *= -1.0;
  }
  result.tail<3>() = x.tail<3>();
  return result;
}
Eigen::Matrix<double, 3, 4> attitude_jacobian(const Eigen::Vector4d &q) {
  Eigen::Matrix<double, 3, 4> result;
  result.col(0) = -2.0 * q.tail<3>();
  result.rightCols<3>() =
      2.0 * (q[0] * Eigen::Matrix3d::Identity() + skew(q.tail<3>()));
  return result;
}
Matrix6x7 state_error_jacobian(const Eigen::Vector4d &q) {
  Matrix6x7 result = Matrix6x7::Zero();
  result.topLeftCorner<3, 4>() = attitude_jacobian(q);
  result.bottomRightCorner<3, 3>().setIdentity();
  return result;
}
Eigen::MatrixXd tangent_covariance(const Eigen::Vector4d &q,
                                   const Eigen::MatrixXd &p) {
  Eigen::Matrix<double, 7, 7> projection =
      Eigen::Matrix<double, 7, 7>::Identity();
  projection.topLeftCorner<4, 4>() -= q * q.transpose();
  const Eigen::Matrix<double, 7, 7> result =
      projection * p * projection.transpose();
  return 0.5 * (result + result.transpose()).eval();
}
void require_seven_state(const State &state) {
  if (state.covariance.rows() != 7 || state.covariance.cols() != 7) {
    throw std::invalid_argument("EKF covariance must be 7x7");
  }
}
Eigen::Vector3d tracker_residual(const Eigen::Vector4d &q,
                                 const Eigen::Vector4d &measured,
                                 const Q &mount) {
  const Q predicted = mount * quat(q);
  Q observation = quat(measured);
  if (observation.coeffs().dot(predicted.coeffs()) < 0.0) {
    observation.coeffs() *= -1.0;
  }
  return logmap(observation * predicted.conjugate());
}
} // namespace

AttitudeEKF::AttitudeEKF(Eigen::Vector4d q, Eigen::Vector3d w,
                         Eigen::MatrixXd p, Eigen::Matrix3d a,
                         std::set<std::string> record)
    : AttitudeEstimator(std::move(q), std::move(w), std::move(p), std::move(a),
                        std::move(record)) {
  if (covariance.rows() != 7) {
    throw std::invalid_argument("EKF covariance must be 7x7");
  }
}

State AttitudeEKF::predict(const State &state,
                           const Eigen::Matrix3d &acceleration_psd, double dt) {
  require_seven_state(state);
  if (!std::isfinite(dt) || dt < 0.0) {
    throw std::invalid_argument("dt must be finite and nonnegative");
  }
  Vector7d x;
  x << state.quaternion, state.rate;
  const Vector7d xp = transition(x, dt);
  Eigen::Matrix<double, 7, 7> f;
  constexpr double epsilon = 1e-7;
  for (Eigen::Index column = 0; column < 7; ++column) {
    Vector7d plus = x, minus = x;
    plus[column] += epsilon;
    minus[column] -= epsilon;
    f.col(column) =
        (transition(plus, dt) - transition(minus, dt)) / (2.0 * epsilon);
  }

  Matrix6d q6 = Matrix6d::Zero();
  q6.topLeftCorner<3, 3>() = acceleration_psd * (dt * dt * dt / 3.0);
  q6.topRightCorner<3, 3>() = -acceleration_psd * (dt * dt / 2.0);
  q6.bottomLeftCorner<3, 3>() = q6.topRightCorner<3, 3>();
  q6.bottomRightCorner<3, 3>() = acceleration_psd * dt;
  Matrix7x6 lift = Matrix7x6::Zero();
  lift.topLeftCorner<1, 3>() = -0.5 * xp.segment<3>(1).transpose();
  lift.block<3, 3>(1, 0) =
      0.5 * (xp[0] * Eigen::Matrix3d::Identity() - skew(xp.segment<3>(1)));
  lift.bottomRightCorner<3, 3>().setIdentity();
  Eigen::Matrix<double, 7, 7> p =
      f * state.covariance * f.transpose() + lift * q6 * lift.transpose();
  p = tangent_covariance(xp.head<4>(), p);
  return {xp.head<4>(), xp.tail<3>(), p};
}

EKFCorrection AttitudeEKF::correct(const State &state,
                                   const Eigen::Vector3d &residual,
                                   const Matrix3x7 &h,
                                   const Eigen::Matrix3d &r) {
  require_seven_state(state);
  const Eigen::Matrix3d s = h * state.covariance * h.transpose() + r;
  const Eigen::LDLT<Eigen::Matrix3d> solver(s);
  if (solver.info() != Eigen::Success ||
      (solver.vectorD().array().abs() < 1e-15).any()) {
    throw std::invalid_argument("innovation covariance is singular");
  }
  const Matrix7x3 gain =
      solver.solve((h * state.covariance).eval()).transpose();
  const Vector7d delta = gain * residual;
  Eigen::Vector4d q = state.quaternion + delta.head<4>();
  if (!q.allFinite() || q.norm() == 0.0) {
    throw std::invalid_argument("EKF quaternion correction is invalid");
  }
  q.normalize();
  const Eigen::Matrix<double, 7, 7> remainder =
      Eigen::Matrix<double, 7, 7>::Identity() - gain * h;
  Eigen::Matrix<double, 7, 7> p =
      remainder * state.covariance * remainder.transpose() +
      gain * r * gain.transpose();
  p = tangent_covariance(q, p);
  State result{q, state.rate + delta.tail<3>(),
               0.5 * (p + p.transpose()).eval()};
  return {result, residual, s, gain, residual - h * delta};
}

EKFCorrection AttitudeEKF::correct_gyro(const State &state,
                                        const Eigen::Vector3d &measured_rate,
                                        const Eigen::Matrix3d &mounting,
                                        const Eigen::Matrix3d &covariance) {
  Matrix3x7 h = Matrix3x7::Zero();
  h.rightCols<3>() = mounting;
  return correct(state, measured_rate - mounting * state.rate, h, covariance);
}

EKFCorrection
AttitudeEKF::correct_star_tracker(const State &state,
                                  const Eigen::Vector4d &measured_quaternion,
                                  const Eigen::Vector4d &mounting_quaternion,
                                  const Eigen::Matrix3d &covariance) {
  require_seven_state(state);
  const Q mount = quat(mounting_quaternion).normalized();
  const Eigen::Vector3d residual =
      tracker_residual(state.quaternion, measured_quaternion, mount);
  Matrix3x7 h = Matrix3x7::Zero();
  constexpr double epsilon = 1e-7;
  for (Eigen::Index column = 0; column < 4; ++column) {
    Eigen::Vector4d plus = state.quaternion, minus = state.quaternion;
    plus[column] += epsilon;
    minus[column] -= epsilon;
    plus.normalize();
    minus.normalize();
    h.col(column) = -(tracker_residual(plus, measured_quaternion, mount) -
                      tracker_residual(minus, measured_quaternion, mount)) /
                    (2.0 * epsilon);
  }
  return correct(state, residual, h, covariance);
}

State AttitudeEKF::step(const State &state,
                        const Eigen::Matrix3d &acceleration_psd, double dt,
                        const Eigen::VectorXd &readings,
                        const std::vector<Sensor> &sensors) {
  State current = predict(state, acceleration_psd, dt);
  for (const Sensor &sensor : sensors) {
    if (!sensor.enabled || (sensor.code != 0 && sensor.code != 2)) {
      continue;
    }
    const Eigen::Index width = sensor.code == 0 ? 3 : 4;
    if (sensor.offset < 0 || sensor.offset + width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const Eigen::VectorXd value = readings.segment(sensor.offset, width);
    if (!value.allFinite()) {
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

std::map<std::string, Eigen::MatrixXd>
AttitudeEKF::sample(const AttitudeEKF &estimator, const Eigen::VectorXd &times,
                    const Eigen::MatrixXd &truth,
                    const Eigen::MatrixXd &readings,
                    const std::vector<Sensor> &sensors, double interval,
                    const std::set<std::string> &fields) {
  if (!std::isfinite(interval) || interval <= 0.0 ||
      truth.rows() != times.size() || truth.cols() < 7 ||
      readings.rows() != times.size()) {
    throw std::invalid_argument(
        "invalid estimator sample trajectory or interval");
  }
  auto output = allocate_samples(times.size(), fields);
  State current = estimator.prepare();
  double previous = 0.0;
  for (Eigen::Index row = 0; row < times.size(); ++row) {
    const double time = times[row];
    if (!std::isfinite(time) || time < 0.0 || (row && time < times[row - 1])) {
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
AttitudeEKF::allocate_samples(Eigen::Index rows, const std::set<std::string> &fields) {
  const std::map<std::string, int> widths{
      {"estimated_attitude", 4},
      {"estimated_rate", 3},
      {"estimated_covariance", 49},
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
    output[name] =
        Eigen::MatrixXd::Constant(rows, widths.at(name),
                                  std::numeric_limits<double>::quiet_NaN());
  }
  return output;
}
void AttitudeEKF::update_sample(State &current,
    const Eigen::Matrix3d &acceleration_psd, double dt,
    const Eigen::VectorXd &truth, const Eigen::VectorXd &readings,
    const std::vector<Sensor> &sensors,
    std::map<std::string, Eigen::MatrixXd> &output, Eigen::Index row) {
  current = predict(current, acceleration_psd, dt);
  auto store = [&](const std::string &name, const auto &value) {
    const auto found = output.find(name);
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
    const auto found = output.find(name);
    if (found != output.end()) {
      found->second(row, 0) = value;
    }
  };
  for (const Sensor &sensor : sensors) {
    if (!sensor.enabled || (sensor.code != 0 && sensor.code != 2)) {
      continue;
    }
    const int width = sensor.code == 0 ? 3 : 4;
    if (sensor.offset < 0 || sensor.offset + width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const Eigen::VectorXd value =
        readings.segment(sensor.offset, width);
    if (!value.allFinite()) {
      continue;
    }
    const Q mount =
        (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
    const Eigen::Matrix3d noise =
        sensor.noise_cholesky * sensor.noise_cholesky.transpose();
    const EKFCorrection correction =
        sensor.code == 0
            ? correct_gyro(current, value.head<3>() - sensor.bias,
                           mount.toRotationMatrix(), noise)
            : correct_star_tracker(current, value.head<4>(), vec(mount),
                                   noise);
    store("measurement_innovation", correction.innovation);
    store("measurement_innovation_covariance",
          correction.innovation_covariance);
    store("measurement_residual", correction.residual);
    const Matrix6x7 tangent = state_error_jacobian(current.quaternion);
    store("kalman_gain", tangent * correction.gain);
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
  const Matrix6x7 error_jacobian = state_error_jacobian(current.quaternion);
  const Matrix6d error_covariance =
      error_jacobian * current.covariance * error_jacobian.transpose();
  store("estimated_attitude_covariance",
        error_covariance.topLeftCorner<3, 3>());
  store("estimated_rate_covariance",
        error_covariance.bottomRightCorner<3, 3>());
  if (output.count("normalized_estimation_error_squared")) {
    Vector6 error;
    error.head<3>() = logmap(quat(truth.head<4>()) *
                             quat(current.quaternion).conjugate());
    error.tail<3>() = truth.segment<3>(4) - current.rate;
    const Eigen::LDLT<Matrix6d> solver(error_covariance);
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array().abs() < 1e-15).any()) {
      throw std::invalid_argument("EKF tangent covariance is singular");
    }
    scalar("normalized_estimation_error_squared",
           error.dot(solver.solve(error)));
  }
}
} // namespace omni::hardware::estimator

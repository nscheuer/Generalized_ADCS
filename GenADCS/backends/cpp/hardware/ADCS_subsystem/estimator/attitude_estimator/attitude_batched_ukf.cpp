#include "attitude_batched_ukf.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omni::hardware::estimator {
namespace {

constexpr int kDim = AttitudeBatchedUKF::COVARIANCE_SIZE;
constexpr int kSigmaCount = 2 * kDim + 1;
constexpr int kStackMaxMeas = 48;
constexpr double kPi = 3.14159265358979323846;
const double kChartLimitAngle = 2.0 * std::asin(0.95);

using Q = Eigen::Quaterniond;
using SigmaWeights = Eigen::Matrix<double, kSigmaCount, 1>;
using SigmaRates = Eigen::Matrix<double, kSigmaCount, 3>;
using SigmaDeviations = Eigen::Matrix<double, kSigmaCount, kDim>;
using SigmaQuats = std::array<Q, kSigmaCount>;

void validate_ukf_params(double alpha, double beta, double kappa) {
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    throw std::invalid_argument("alpha must be finite and positive");
  }
  if (!std::isfinite(beta)) {
    throw std::invalid_argument("beta must be finite");
  }
  if (!std::isfinite(kappa)) {
    throw std::invalid_argument("kappa must be finite");
  }
  if (alpha * alpha * (kDim + kappa) <= 0.0) {
    throw std::invalid_argument("alpha and kappa must give a positive UKF scale");
  }
}

inline Q quat(const Eigen::Vector4d &v) {
  return Q(v[0], v[1], v[2], v[3]);
}

inline Eigen::Vector4d vec(Q q) {
  q.normalize();
  return {q.w(), q.x(), q.y(), q.z()};
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

inline Q expmap(const Eigen::Vector3d &v) {
  const double angle = v.norm();
  if (angle == 0.0) {
    return Q(1.0, 0.0, 0.0, 0.0);
  }
  if (angle < 1e-12) {
    return Q(1.0, 0.5 * v.x(), 0.5 * v.y(), 0.5 * v.z()).normalized();
  }
  const double half = 0.5 * angle;
  const double scale = std::sin(half) / angle;
  return Q(std::cos(half), v.x() * scale, v.y() * scale, v.z() * scale);
}

inline Eigen::Vector3d logmap(Q q) {
  q.normalize();
  if (q.w() < 0.0) {
    q.coeffs() *= -1.0;
  }
  const double n = q.vec().norm();
  if (n == 0.0) {
    return Eigen::Vector3d::Zero();
  }
  if (n < 1e-12) {
    return 2.0 * q.vec();
  }
  return (2.0 * std::atan2(n, q.w()) / n) * q.vec();
}

inline double wrap_angle(double value) {
  constexpr double two_pi = 2.0 * kPi;
  return value - two_pi * std::floor((value + kPi) / two_pi);
}

inline void ukf_weights(double alpha, double beta, double kappa, double &gamma,
                        SigmaWeights &w_m, SigmaWeights &w_c) {
  validate_ukf_params(alpha, beta, kappa);
  const double scale = alpha * alpha * (kDim + kappa);
  const double lam = scale - static_cast<double>(kDim);
  gamma = std::sqrt(scale);
  w_m.setConstant(0.5 / scale);
  w_c.setConstant(0.5 / scale);
  w_m[0] = lam / scale;
  w_c[0] = lam / scale + (1.0 - alpha * alpha + beta);
}

inline Matrix6 cholesky_psd(const Matrix6 &p) {
  const Matrix6 p_sym = 0.5 * (p + p.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix6> es(p_sym);
  if (es.eigenvalues()[0] > 1.0e-15) {
    Eigen::LLT<Matrix6> llt(p_sym);
    if (llt.info() == Eigen::Success) {
      return llt.matrixL();
    }
  }
  const Vector6 vals = es.eigenvalues().cwiseMax(0.0);
  const Matrix6 rec =
      es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose() +
      1.0e-14 * Matrix6::Identity();
  const Matrix6 rec_sym = 0.5 * (rec + rec.transpose());
  Eigen::LLT<Matrix6> llt(rec_sym);
  return llt.matrixL();
}

Q quaternion_mean(const SigmaQuats &quaternions, const SigmaWeights &weights) {
  SigmaQuats normalized;
  for (int i = 0; i < kSigmaCount; ++i) {
    normalized[i] = quaternions[i].normalized();
  }
  Q mean = normalized[0];
  std::array<Eigen::Vector3d, kSigmaCount> deviations;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (int iter = 0; iter < 32; ++iter) {
    const Q mean_conj = mean.conjugate();
    double scale = 0.0;
    Eigen::Vector3d residual = Eigen::Vector3d::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
      deviations[i] = logmap(normalized[i] * mean_conj);
      scale += std::abs(weights[i]) * deviations[i].norm();
      residual += weights[i] * deviations[i];
    }
    const double residual_norm = residual.norm();
    const double tolerance = 1.0e-12 + 64.0 * eps * scale;
    if (residual_norm <= tolerance) {
      return mean;
    }
    const Eigen::Vector3d step =
        residual * std::min(1.0, (0.5 * kPi) / std::max(residual_norm, 1.0e-15));
    bool improved = false;
    double step_alpha = 1.0;
    for (int ls = 0; ls < 16; ++ls) {
      const Q candidate = (expmap(step_alpha * step) * mean).normalized();
      const Q cand_conj = candidate.conjugate();
      Eigen::Vector3d cand_residual = Eigen::Vector3d::Zero();
      for (int i = 0; i < kSigmaCount; ++i) {
        cand_residual += weights[i] * logmap(normalized[i] * cand_conj);
      }
      if (cand_residual.norm() < residual_norm) {
        mean = candidate;
        improved = true;
        break;
      }
      step_alpha *= 0.5;
    }
    if (!improved) {
      break;
    }
  }
  return mean;
}

void sigma_points(const Q &q, const Eigen::Vector3d &w, const Matrix6 &p,
                  double alpha, double beta, double kappa, SigmaQuats &q_pts,
                  SigmaRates &w_pts, SigmaWeights &w_m, SigmaWeights &w_c) {
  double gamma = 0.0;
  ukf_weights(alpha, beta, kappa, gamma, w_m, w_c);
  const Matrix6 chol = cholesky_psd(p);
  q_pts[0] = q.normalized();
  w_pts.row(0) = w.transpose();
  for (int i = 0; i < kDim; ++i) {
    const Vector6 offset = gamma * chol.col(i);
    Eigen::Vector3d d_theta = offset.head<3>();
    const double theta_norm = d_theta.norm();
    if (theta_norm > kChartLimitAngle) {
      d_theta *= (kChartLimitAngle / theta_norm);
    }
    const Q dq = expmap(d_theta);
    q_pts[1 + i] = (dq * q_pts[0]).normalized();
    w_pts.row(1 + i) = (w + offset.tail<3>()).transpose();
    q_pts[1 + kDim + i] = (dq.conjugate() * q_pts[0]).normalized();
    w_pts.row(1 + kDim + i) = (w - offset.tail<3>()).transpose();
  }
}

void state_deviations(const SigmaQuats &q_pts, const SigmaRates &w_pts,
                      const Q &q_ref, const Eigen::Vector3d &w_ref,
                      SigmaDeviations &deviations) {
  const Q q_ref_conj = q_ref.conjugate();
  for (int k = 0; k < kSigmaCount; ++k) {
    deviations.row(k).head<3>() = logmap(q_pts[k] * q_ref_conj).transpose();
    deviations.row(k).tail<3>() = w_pts.row(k) - w_ref.transpose();
  }
}

void predict_fixed(Q &q, Eigen::Vector3d &w, Matrix6 &p,
                   const Eigen::Matrix3d &acceleration_psd, double dt,
                   double alpha, double beta, double kappa) {
  if (!std::isfinite(dt) || dt < 0.0) {
    throw std::invalid_argument("dt must be finite and nonnegative");
  }
  SigmaQuats q_pts;
  SigmaRates w_pts;
  SigmaWeights w_m, w_c;
  sigma_points(q, w, p, alpha, beta, kappa, q_pts, w_pts, w_m, w_c);

  SigmaQuats q_prop;
  for (int k = 0; k < kSigmaCount; ++k) {
    const Eigen::Vector3d w_k = w_pts.row(k).transpose();
    const Q increment = expmap(-w_k * dt);
    q_prop[k] = (increment * q_pts[k]).normalized();
  }

  const Q q_mean = quaternion_mean(q_prop, w_m);
  const Eigen::Vector3d w_mean = (w_m.transpose() * w_pts).transpose();

  SigmaDeviations deviations;
  state_deviations(q_prop, w_pts, q_mean, w_mean, deviations);

  Matrix6 noise = Matrix6::Zero();
  noise.topLeftCorner<3, 3>() = acceleration_psd * (dt * dt * dt / 3.0);
  noise.topRightCorner<3, 3>() = -acceleration_psd * (dt * dt / 2.0);
  noise.bottomLeftCorner<3, 3>() = noise.topRightCorner<3, 3>();
  noise.bottomRightCorner<3, 3>() = acceleration_psd * dt;

  const Matrix6 p_pred =
      deviations.transpose() * w_c.asDiagonal() * deviations + noise;
  q = q_mean;
  w = w_mean;
  p = 0.5 * (p_pred + p_pred.transpose());
}

template <int M>
inline Eigen::Matrix<double, M, M> sensor_noise_matrix(const Sensor &sensor) {
  if constexpr (M == 1) {
    Eigen::Matrix<double, 1, 1> r;
    r(0, 0) = sensor.noise_std * sensor.noise_std;
    return r;
  } else {
    return sensor.noise_cholesky.topLeftCorner<M, M>() *
           sensor.noise_cholesky.topLeftCorner<M, M>().transpose();
  }
}

template <int M>
inline bool direction_measurement_fixed(const Sensor &sensor,
                                        const Eigen::Matrix3d &mount_rot,
                                        const Q &attitude,
                                        Eigen::Matrix<double, M, 1> &out) {
  const Eigen::Vector3d direction =
      mount_rot * (attitude.normalized().toRotationMatrix() * sensor.reference);
  if constexpr (M == 3) {
    out = (direction + sensor.bias)
              .cwiseMax(-sensor.saturation)
              .cwiseMin(sensor.saturation);
    return out.allFinite();
  } else if constexpr (M == 1) {
    out[0] = std::clamp(sensor.sensitivity *
                                std::max(0.0, sensor.normal.dot(direction)) +
                            sensor.bias[0],
                        0.0, sensor.saturation);
    return std::isfinite(out[0]);
  } else {
    const double angle_x = std::atan2(direction[0], direction[2]);
    const double angle_y = std::atan2(direction[1], direction[2]);
    if (std::abs(angle_x) > sensor.fov[0] ||
        std::abs(angle_y) > sensor.fov[1]) {
      return false;
    }
    out[0] = angle_x + sensor.bias[0];
    out[1] = angle_y + sensor.bias[1];
    return out.allFinite();
  }
}

template <int M, typename DzBuf, typename ResBuf, typename RBuf>
bool append_direction_sensor(const Sensor &sensor,
                             const Eigen::Matrix<double, M, 1> &observed,
                             const SigmaQuats &q_pts, const SigmaWeights &w_m,
                             DzBuf &dz_buf, ResBuf &res_buf, RBuf &r_buf,
                             Eigen::Index &m_total) {
  const Q mount_q =
      (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
  const Eigen::Matrix3d mount_rot = mount_q.toRotationMatrix();

  Eigen::Matrix<double, kSigmaCount, M> z_pts;
  for (int k = 0; k < kSigmaCount; ++k) {
    Eigen::Matrix<double, M, 1> z_k;
    if (!direction_measurement_fixed<M>(sensor, mount_rot, q_pts[k], z_k)) {
      return false;
    }
    z_pts.row(k) = z_k.transpose();
  }

  if (sensor.code == 4 || sensor.code == 5) {
    Eigen::Matrix<double, kSigmaCount, M> wrapped_deltas;
    for (int k = 0; k < kSigmaCount; ++k) {
      for (int j = 0; j < M; ++j) {
        wrapped_deltas(k, j) = wrap_angle(z_pts(k, j) - z_pts(0, j));
      }
    }
    const Eigen::Matrix<double, M, 1> delta_mean =
        (w_m.transpose() * wrapped_deltas).transpose();
    Eigen::Matrix<double, M, 1> z_mean;
    for (int j = 0; j < M; ++j) {
      z_mean[j] = wrap_angle(z_pts(0, j) + delta_mean[j]);
      res_buf(m_total + j) = wrap_angle(observed[j] - z_mean[j]);
    }
    for (int k = 0; k < kSigmaCount; ++k) {
      for (int j = 0; j < M; ++j) {
        dz_buf(m_total + j, k) = wrap_angle(z_pts(k, j) - z_mean[j]);
      }
    }
  } else {
    const Eigen::Matrix<double, M, 1> z_mean =
        (w_m.transpose() * z_pts).transpose();
    dz_buf.template middleRows<M>(m_total) =
        (z_pts.rowwise() - z_mean.transpose()).transpose();
    res_buf.template segment<M>(m_total) = observed - z_mean;
  }

  r_buf.template block<M, M>(m_total, m_total) = sensor_noise_matrix<M>(sensor);
  m_total += M;
  return true;
}

template <int MaxMeas>
void step_impl_capacity(Q &q, Eigen::Vector3d &w, Matrix6 &p,
                        const Eigen::Matrix3d &acceleration_psd, double dt,
                        const Eigen::VectorXd &readings,
                        const std::vector<Sensor> &sensors, double alpha,
                        double beta, double kappa,
                        std::map<std::string, Eigen::MatrixXd> *output,
                        Eigen::Index row, Eigen::Index max_meas) {
  predict_fixed(q, w, p, acceleration_psd, dt, alpha, beta, kappa);

  if (sensors.empty()) {
    return;
  }

  SigmaQuats q_pts;
  SigmaRates w_pts;
  SigmaWeights w_m, w_c;
  sigma_points(q, w, p, alpha, beta, kappa, q_pts, w_pts, w_m, w_c);
  SigmaDeviations dx_pts;
  state_deviations(q_pts, w_pts, q, w, dx_pts);

  using DzBuf = Eigen::Matrix<double, Eigen::Dynamic, kSigmaCount,
                              Eigen::ColMajor, MaxMeas, kSigmaCount>;
  using ResBuf =
      Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, MaxMeas, 1>;
  using MeasMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::ColMajor, MaxMeas, MaxMeas>;
  using CrossMat = Eigen::Matrix<double, kDim, Eigen::Dynamic, Eigen::ColMajor,
                                 kDim, MaxMeas>;

  DzBuf dz_buf(max_meas, kSigmaCount);
  ResBuf res_buf(max_meas);
  MeasMat r_buf = MeasMat::Zero(max_meas, max_meas);
  Eigen::Index m_total = 0;

  for (const Sensor &sensor : sensors) {
    if (!sensor.enabled || sensor.code < 0 || sensor.code > 5) {
      continue;
    }
    const Eigen::Index obs_width =
        sensor.code == 2                         ? 4
        : (sensor.code == 0 || sensor.code == 1) ? 3
        : sensor.code == 3                       ? 1
                                                 : 2;
    if (sensor.offset < 0 || sensor.offset + obs_width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const auto observed_seg = readings.segment(sensor.offset, obs_width);
    if (!observed_seg.allFinite()) {
      continue;
    }

    if (sensor.code == 0) {
      const Eigen::Vector3d observed = observed_seg;
      const Q mount_q =
          (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
      const Eigen::Matrix3d mount = mount_q.toRotationMatrix();

      Eigen::Matrix<double, kSigmaCount, 3> z_pts;
      for (int k = 0; k < kSigmaCount; ++k) {
        z_pts.row(k) =
            (mount * w_pts.row(k).transpose() + sensor.bias).transpose();
      }
      const Eigen::Vector3d z_mean = (w_m.transpose() * z_pts).transpose();
      dz_buf.template middleRows<3>(m_total) =
          (z_pts.rowwise() - z_mean.transpose()).transpose();
      res_buf.template segment<3>(m_total) = observed - z_mean;
      r_buf.template block<3, 3>(m_total, m_total) =
          sensor_noise_matrix<3>(sensor);
      m_total += 3;
    } else if (sensor.code == 2) {
      const Eigen::Vector4d observed = observed_seg;
      const Q mount_q =
          (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();

      SigmaQuats z_pts;
      for (int k = 0; k < kSigmaCount; ++k) {
        z_pts[k] = (mount_q * q_pts[k]).normalized();
      }
      const Q z_mean = quaternion_mean(z_pts, w_m);
      const Q z_mean_conj = z_mean.conjugate();
      for (int k = 0; k < kSigmaCount; ++k) {
        dz_buf.template block<3, 1>(m_total, k) =
            logmap(z_pts[k] * z_mean_conj);
      }
      const Q obs_q = quat(observed).normalized();
      res_buf.template segment<3>(m_total) = logmap(obs_q * z_mean_conj);
      r_buf.template block<3, 3>(m_total, m_total) =
          sensor_noise_matrix<3>(sensor);
      m_total += 3;
    } else if (sensor.code == 1) {
      const Eigen::Vector3d observed = observed_seg;
      append_direction_sensor<3>(sensor, observed, q_pts, w_m, dz_buf, res_buf,
                                 r_buf, m_total);
    } else if (sensor.code == 3) {
      const Eigen::Matrix<double, 1, 1> observed = observed_seg;
      append_direction_sensor<1>(sensor, observed, q_pts, w_m, dz_buf, res_buf,
                                 r_buf, m_total);
    } else {
      const Eigen::Vector2d observed = observed_seg;
      append_direction_sensor<2>(sensor, observed, q_pts, w_m, dz_buf, res_buf,
                                 r_buf, m_total);
    }
  }

  if (m_total > 0) {
    const auto dz_t = dz_buf.topRows(m_total);
    const auto res_all = res_buf.head(m_total);
    const auto r_all = r_buf.topLeftCorner(m_total, m_total);

    MeasMat p_zz = dz_t * w_c.asDiagonal() * dz_t.transpose() + r_all;
    p_zz = 0.5 * (p_zz + p_zz.transpose());
    const CrossMat p_xz = dx_pts.transpose() * w_c.asDiagonal() * dz_t.transpose();

    const auto solver = p_zz.ldlt();
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array().abs() < 1e-20).any()) {
      throw std::invalid_argument("innovation covariance is singular");
    }
    const CrossMat gain = solver.solve(p_xz.transpose()).transpose();
    const Vector6 delta = gain * res_all;
    const Matrix6 p_prior = p;

    q = (expmap(delta.head<3>()) * q).normalized();
    w = w + delta.tail<3>();

    const Matrix6 p_raw = p - gain * p_zz * gain.transpose();
    Matrix6 reset = Matrix6::Identity();
    reset.topLeftCorner<3, 3>() += 0.5 * skew(delta.head<3>());
    const Matrix6 p_next = reset * p_raw * reset.transpose();
    p = 0.5 * (p_next + p_next.transpose());

    if (output != nullptr) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const Eigen::Index diag_w = std::min<Eigen::Index>(3, m_total);
      if (auto it = output->find("measurement_innovation");
          it != output->end()) {
        it->second.row(row).setConstant(nan);
        it->second.row(row).head(diag_w) = res_all.head(diag_w).transpose();
      }
      if (auto it = output->find("measurement_innovation_covariance");
          it != output->end()) {
        Eigen::Matrix3d block = Eigen::Matrix3d::Constant(nan);
        block.topLeftCorner(diag_w, diag_w) =
            p_zz.topLeftCorner(diag_w, diag_w);
        Eigen::Index idx = 0;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            it->second(row, idx++) = block(i, j);
          }
        }
      }
      if (auto it = output->find("normalized_innovation_squared");
          it != output->end()) {
        it->second(row, 0) = res_all.dot(solver.solve(res_all));
      }
      if (auto it = output->find("measurement_residual"); it != output->end()) {
        const auto prior_solver = p_prior.ldlt();
        if (prior_solver.info() != Eigen::Success ||
            (prior_solver.vectorD().array().abs() < 1e-20).any()) {
          throw std::invalid_argument("prior covariance is singular");
        }
        const auto h_eff = prior_solver.solve(p_xz).transpose();
        const ResBuf post_res = res_all - h_eff * delta;
        it->second.row(row).setConstant(nan);
        it->second.row(row).head(diag_w) = post_res.head(diag_w).transpose();
      }
      if (auto it = output->find("kalman_gain"); it != output->end()) {
        Eigen::Matrix<double, kDim, 3> block =
            Eigen::Matrix<double, kDim, 3>::Constant(nan);
        block.leftCols(diag_w) = gain.leftCols(diag_w);
        Eigen::Index idx = 0;
        for (int i = 0; i < kDim; ++i) {
          for (int j = 0; j < 3; ++j) {
            it->second(row, idx++) = block(i, j);
          }
        }
      }
    }
  }
}

void step_impl(Q &q, Eigen::Vector3d &w, Matrix6 &p,
               const Eigen::Matrix3d &acceleration_psd, double dt,
               const Eigen::VectorXd &readings,
               const std::vector<Sensor> &sensors, double alpha, double beta,
               double kappa, std::map<std::string, Eigen::MatrixXd> *output,
               Eigen::Index row) {
  const Eigen::Index max_meas = 3 * static_cast<Eigen::Index>(sensors.size());
  if (max_meas <= kStackMaxMeas) {
    step_impl_capacity<kStackMaxMeas>(q, w, p, acceleration_psd, dt, readings,
                                      sensors, alpha, beta, kappa, output, row,
                                      max_meas);
  } else {
    step_impl_capacity<Eigen::Dynamic>(q, w, p, acceleration_psd, dt, readings,
                                       sensors, alpha, beta, kappa, output, row,
                                       max_meas);
  }
}

} // namespace

AttitudeBatchedUKF::AttitudeBatchedUKF(Eigen::Vector4d quaternion_in,
                                       Eigen::Vector3d rate_in,
                                       Eigen::MatrixXd covariance_in,
                                       Eigen::Matrix3d acceleration_psd_in,
                                       double alpha_in, double beta_in,
                                       double kappa_in,
                                       std::set<std::string> record_in)
    : AttitudeEstimator(std::move(quaternion_in), std::move(rate_in),
                        std::move(covariance_in),
                        std::move(acceleration_psd_in), std::move(record_in)),
      alpha(alpha_in), beta(beta_in), kappa(kappa_in),
      ukf_params(alpha_in, beta_in, kappa_in) {
  if (covariance.rows() != COVARIANCE_SIZE) {
    throw std::invalid_argument("UKF covariance must be 6x6");
  }
  validate_ukf_params(alpha, beta, kappa);
}

State AttitudeBatchedUKF::predict(const State &state,
                                  const Eigen::Matrix3d &acceleration_psd,
                                  double dt, double alpha, double beta,
                                  double kappa) {
  if (state.covariance.rows() != COVARIANCE_SIZE ||
      state.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("UKF covariance must be 6x6");
  }
  Q q = quat(state.quaternion);
  Eigen::Vector3d w = state.rate;
  Matrix6 p = state.covariance;
  predict_fixed(q, w, p, acceleration_psd, dt, alpha, beta, kappa);
  return {vec(q), w, p};
}

State AttitudeBatchedUKF::step(const State &state,
                               const Eigen::Matrix3d &acceleration_psd,
                               double dt, const Eigen::VectorXd &readings,
                               const std::vector<Sensor> &sensors,
                               double alpha, double beta, double kappa) {
  if (state.covariance.rows() != COVARIANCE_SIZE ||
      state.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("UKF covariance must be 6x6");
  }
  Q q = quat(state.quaternion);
  Eigen::Vector3d w = state.rate;
  Matrix6 p = state.covariance;
  step_impl(q, w, p, acceleration_psd, dt, readings, sensors, alpha, beta,
            kappa, nullptr, 0);
  return {vec(q), w, p};
}

std::map<std::string, Eigen::MatrixXd>
AttitudeBatchedUKF::allocate_samples(Eigen::Index rows,
                                     const std::set<std::string> &fields) {
  return AttitudeMEKF::allocate_samples(rows, fields);
}

void AttitudeBatchedUKF::update_sample(
    State &current, const Eigen::Matrix3d &acceleration_psd, double dt,
    const Eigen::VectorXd &truth, const Eigen::VectorXd &readings,
    const std::vector<Sensor> &sensors,
    std::map<std::string, Eigen::MatrixXd> &output, Eigen::Index row,
    double alpha, double beta, double kappa) {
  if (current.covariance.rows() != COVARIANCE_SIZE ||
      current.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("UKF covariance must be 6x6");
  }
  Q q = quat(current.quaternion);
  Eigen::Vector3d w = current.rate;
  Matrix6 p = current.covariance;
  step_impl(q, w, p, acceleration_psd, dt, readings, sensors, alpha, beta,
            kappa, &output, row);

  current.quaternion = vec(q);
  current.rate = w;
  current.covariance = p;

  if (auto it = output.find("estimated_attitude"); it != output.end()) {
    it->second.row(row) = current.quaternion.transpose();
  }
  if (auto it = output.find("estimated_rate"); it != output.end()) {
    it->second.row(row) = current.rate.transpose();
  }
  if (auto it = output.find("estimated_covariance"); it != output.end()) {
    Eigen::Index idx = 0;
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        it->second(row, idx++) = p(i, j);
      }
    }
  }
  if (auto it = output.find("estimated_attitude_covariance");
      it != output.end()) {
    Eigen::Index idx = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        it->second(row, idx++) = p(i, j);
      }
    }
  }
  if (auto it = output.find("estimated_rate_covariance"); it != output.end()) {
    Eigen::Index idx = 0;
    for (int i = 3; i < kDim; ++i) {
      for (int j = 3; j < kDim; ++j) {
        it->second(row, idx++) = p(i, j);
      }
    }
  }
  if (auto it = output.find("normalized_estimation_error_squared");
      it != output.end()) {
    Vector6 error;
    error.head<3>() = logmap(quat(truth.head<4>()) * q.conjugate());
    error.tail<3>() = truth.segment<3>(4) - w;
    const auto solver = p.ldlt();
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array() == 0.0).any()) {
      throw std::invalid_argument("estimator covariance is singular");
    }
    it->second(row, 0) = error.dot(solver.solve(error));
  }
}

std::map<std::string, Eigen::MatrixXd> AttitudeBatchedUKF::sample(
    const AttitudeBatchedUKF &estimator, const Eigen::VectorXd &times,
    const Eigen::MatrixXd &truth, const Eigen::MatrixXd &readings,
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
                  sensors, output, row, estimator.alpha, estimator.beta,
                  estimator.kappa);
    previous = time;
  }
  return output;
}

} // namespace omni::hardware::estimator

#include "attitude_sequential_srukf.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omni::hardware::estimator {
namespace {

constexpr int kDim = AttitudeSequentialSRUKF::COVARIANCE_SIZE;
constexpr int kSigmaCount = 2 * kDim + 1;
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

template <int N>
inline Eigen::Matrix<double, N, N>
cholesky_psd(const Eigen::Matrix<double, N, N> &p) {
  using MatN = Eigen::Matrix<double, N, N>;
  using VecN = Eigen::Matrix<double, N, 1>;
  if constexpr (N == 1) {
    MatN out;
    out(0, 0) = p(0, 0) > 1.0e-15 ? std::sqrt(p(0, 0))
                                  : std::sqrt(std::max(p(0, 0), 0.0) + 1.0e-14);
    return out;
  } else {
    const MatN p_sym = 0.5 * (p + p.transpose());
    Eigen::SelfAdjointEigenSolver<MatN> es(p_sym);
    if (es.eigenvalues()[0] > 1.0e-15) {
      Eigen::LLT<MatN> llt(p_sym);
      if (llt.info() == Eigen::Success) {
        return llt.matrixL();
      }
    }
    const VecN vals = es.eigenvalues().cwiseMax(0.0);
    const MatN rec =
        es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose() +
        1.0e-14 * MatN::Identity();
    const MatN rec_sym = 0.5 * (rec + rec.transpose());
    Eigen::LLT<MatN> llt(rec_sym);
    return llt.matrixL();
  }
}

template <int N>
inline Eigen::Matrix<double, N, N>
cholupdate(const Eigen::Matrix<double, N, N> &s_mat,
           const Eigen::Matrix<double, N, 1> &vec, double sign) {
  Eigen::Matrix<double, N, N> out = s_mat;
  Eigen::Matrix<double, N, 1> x = vec;
  for (int k = 0; k < N; ++k) {
    const double diag = out(k, k);
    if (std::abs(diag) < 1.0e-15) {
      return cholesky_psd<N>(s_mat * s_mat.transpose() +
                             sign * (vec * vec.transpose()));
    }
    const double r2 = diag * diag + sign * x[k] * x[k];
    if (r2 <= 1.0e-30) {
      return cholesky_psd<N>(s_mat * s_mat.transpose() +
                             sign * (vec * vec.transpose()));
    }
    const double r = std::sqrt(r2);
    const double c = r / diag;
    const double s = x[k] / diag;
    out(k, k) = r;
    for (int j = k + 1; j < N; ++j) {
      out(j, k) = (out(j, k) + sign * s * x[j]) / c;
      x[j] = c * x[j] - s * out(j, k);
    }
  }
  return out;
}

template <int Rows, int Cols>
inline Eigen::Matrix<double, Cols, Cols>
qr_lower_triangular(const Eigen::Matrix<double, Rows, Cols> &compound) {
  Eigen::HouseholderQR<Eigen::Matrix<double, Rows, Cols>> qr(compound);
  const Eigen::Matrix<double, Cols, Cols> r =
      qr.matrixQR()
          .template topLeftCorner<Cols, Cols>()
          .template triangularView<Eigen::Upper>();
  Eigen::Matrix<double, Cols, Cols> s = r.transpose();
  for (int i = 0; i < Cols; ++i) {
    if (s(i, i) < 0.0) {
      s.col(i) = -s.col(i);
    }
  }
  return s;
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

void sigma_points_from_sqrt(const Q &q, const Eigen::Vector3d &w,
                            const Matrix6 &s_mat, double alpha, double beta,
                            double kappa, SigmaQuats &q_pts, SigmaRates &w_pts,
                            SigmaWeights &w_m, SigmaWeights &w_c) {
  double gamma = 0.0;
  ukf_weights(alpha, beta, kappa, gamma, w_m, w_c);
  q_pts[0] = q.normalized();
  w_pts.row(0) = w.transpose();
  for (int i = 0; i < kDim; ++i) {
    const Vector6 offset = gamma * s_mat.col(i);
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

void predict_sqrt(Q &q, Eigen::Vector3d &w, Matrix6 &s_mat,
                  const Eigen::Matrix3d &acceleration_psd, double dt,
                  double alpha, double beta, double kappa) {
  if (!std::isfinite(dt) || dt < 0.0) {
    throw std::invalid_argument("dt must be finite and nonnegative");
  }
  SigmaQuats q_pts;
  SigmaRates w_pts;
  SigmaWeights w_m, w_c;
  sigma_points_from_sqrt(q, w, s_mat, alpha, beta, kappa, q_pts, w_pts, w_m,
                         w_c);

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

  const bool has_noise = dt > 0.0 && noise.cwiseAbs().maxCoeff() > 0.0;
  Matrix6 s_pred;
  if (has_noise) {
    Eigen::Matrix<double, kSigmaCount - 1 + kDim, kDim> compound;
    for (int k = 1; k < kSigmaCount; ++k) {
      compound.row(k - 1) = std::sqrt(w_c[k]) * deviations.row(k);
    }
    const Matrix6 s_q = cholesky_psd<kDim>(noise);
    compound.bottomRows<kDim>() = s_q.transpose();
    s_pred = qr_lower_triangular<kSigmaCount - 1 + kDim, kDim>(compound);
  } else {
    Eigen::Matrix<double, kSigmaCount - 1, kDim> compound;
    for (int k = 1; k < kSigmaCount; ++k) {
      compound.row(k - 1) = std::sqrt(w_c[k]) * deviations.row(k);
    }
    s_pred = qr_lower_triangular<kSigmaCount - 1, kDim>(compound);
  }

  const double sign0 = w_c[0] >= 0.0 ? 1.0 : -1.0;
  s_mat = cholupdate<kDim>(
      s_pred, std::sqrt(std::abs(w_c[0])) * deviations.row(0).transpose(),
      sign0);
  q = q_mean;
  w = w_mean;
}

template <int M>
inline Eigen::Matrix<double, M, M> sensor_noise_sqrt(const Sensor &sensor) {
  if constexpr (M == 1) {
    Eigen::Matrix<double, 1, 1> s_r;
    s_r(0, 0) = std::abs(sensor.noise_std);
    return s_r;
  } else {
    return sensor.noise_cholesky.topLeftCorner<M, M>();
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

template <int M>
void correct_srukf(Q &q, Eigen::Vector3d &w, Matrix6 &s_mat,
                   const SigmaDeviations &dx_pts,
                   const Eigen::Matrix<double, kSigmaCount, M> &dz_pts,
                   const SigmaWeights &w_c,
                   const Eigen::Matrix<double, M, M> &s_r,
                   const Eigen::Matrix<double, M, 1> &residual,
                   Eigen::Matrix<double, M, M> &innovation_out,
                   Eigen::Matrix<double, kDim, M> &gain_out) {
  Eigen::Matrix<double, kSigmaCount - 1 + M, M> compound;
  for (int k = 1; k < kSigmaCount; ++k) {
    compound.row(k - 1) = std::sqrt(w_c[k]) * dz_pts.row(k);
  }
  compound.template bottomRows<M>() = s_r.transpose();
  Eigen::Matrix<double, M, M> s_z =
      qr_lower_triangular<kSigmaCount - 1 + M, M>(compound);

  const double sign0 = w_c[0] >= 0.0 ? 1.0 : -1.0;
  s_z = cholupdate<M>(
      s_z, std::sqrt(std::abs(w_c[0])) * dz_pts.row(0).transpose(), sign0);

  if (!s_z.allFinite() || (s_z.diagonal().array().abs() < 1e-15).any()) {
    throw std::invalid_argument("innovation covariance is singular");
  }

  const Eigen::Matrix<double, kDim, M> p_xz =
      dx_pts.transpose() * w_c.asDiagonal() * dz_pts;

  const Eigen::Matrix<double, M, kDim> y_solve =
      s_z.template triangularView<Eigen::Lower>().solve(p_xz.transpose());
  const Eigen::Matrix<double, M, kDim> k_t =
      s_z.transpose().template triangularView<Eigen::Upper>().solve(y_solve);
  const Eigen::Matrix<double, kDim, M> gain = k_t.transpose();

  const Vector6 delta = gain * residual;
  const Q q_next = (expmap(delta.head<3>()) * q).normalized();
  const Eigen::Vector3d w_next = w + delta.tail<3>();

  const Eigen::Matrix<double, kDim, M> u_mat = gain * s_z;
  Matrix6 s_raw = s_mat;
  for (int j = 0; j < M; ++j) {
    s_raw = cholupdate<kDim>(s_raw, u_mat.col(j), -1.0);
  }

  Matrix6 reset = Matrix6::Identity();
  reset.topLeftCorner<3, 3>() += 0.5 * skew(delta.head<3>());
  const Matrix6 s_next =
      qr_lower_triangular<kDim, kDim>((reset * s_raw).transpose());

  Eigen::Matrix<double, M, M> innovation = s_z * s_z.transpose();
  innovation = 0.5 * (innovation + innovation.transpose());

  q = q_next;
  w = w_next;
  s_mat = s_next;
  innovation_out = innovation;
  gain_out = gain;
}

template <int M>
void record_measurement_diagnostics(
    std::map<std::string, Eigen::MatrixXd> &output, Eigen::Index row,
    const Eigen::Matrix<double, M, 1> &residual,
    const Eigen::Matrix<double, M, M> &innovation,
    const Eigen::Matrix<double, M, 1> &post_residual,
    const Eigen::Matrix<double, kDim, M> &gain) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (auto it = output.find("measurement_innovation"); it != output.end()) {
    it->second.row(row).setConstant(nan);
    it->second.row(row).head<M>() = residual.transpose();
  }
  if (auto it = output.find("measurement_innovation_covariance");
      it != output.end()) {
    Eigen::Matrix3d block = Eigen::Matrix3d::Constant(nan);
    block.topLeftCorner<M, M>() = innovation;
    Eigen::Index idx = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        it->second(row, idx++) = block(i, j);
      }
    }
  }
  if (auto it = output.find("normalized_innovation_squared");
      it != output.end()) {
    it->second(row, 0) = residual.dot(innovation.ldlt().solve(residual));
  }
  if (auto it = output.find("measurement_residual"); it != output.end()) {
    it->second.row(row).setConstant(nan);
    it->second.row(row).head<M>() = post_residual.transpose();
  }
  if (auto it = output.find("kalman_gain"); it != output.end()) {
    Eigen::Matrix<double, kDim, 3> block =
        Eigen::Matrix<double, kDim, 3>::Constant(nan);
    block.leftCols<M>() = gain;
    Eigen::Index idx = 0;
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < 3; ++j) {
        it->second(row, idx++) = block(i, j);
      }
    }
  }
}

template <int M>
void process_direction_sensor(
    Q &q, Eigen::Vector3d &w, Matrix6 &s_mat, const Sensor &sensor,
    const Eigen::Matrix<double, M, 1> &observed, const SigmaQuats &q_pts,
    const SigmaDeviations &dx_pts, const SigmaWeights &w_m,
    const SigmaWeights &w_c, std::map<std::string, Eigen::MatrixXd> *output,
    Eigen::Index row) {
  const Q mount_q =
      (quat(sensor.misalignment) * quat(sensor.orientation)).normalized();
  const Eigen::Matrix3d mount_rot = mount_q.toRotationMatrix();

  Eigen::Matrix<double, kSigmaCount, M> z_pts;
  for (int k = 0; k < kSigmaCount; ++k) {
    Eigen::Matrix<double, M, 1> z_k;
    if (!direction_measurement_fixed<M>(sensor, mount_rot, q_pts[k], z_k)) {
      return;
    }
    z_pts.row(k) = z_k.transpose();
  }

  Eigen::Matrix<double, M, 1> z_mean;
  Eigen::Matrix<double, kSigmaCount, M> dz_pts;
  Eigen::Matrix<double, M, 1> residual;
  if (sensor.code == 4 || sensor.code == 5) {
    Eigen::Matrix<double, kSigmaCount, M> wrapped_deltas;
    for (int k = 0; k < kSigmaCount; ++k) {
      for (int j = 0; j < M; ++j) {
        wrapped_deltas(k, j) = wrap_angle(z_pts(k, j) - z_pts(0, j));
      }
    }
    const Eigen::Matrix<double, M, 1> delta_mean =
        (w_m.transpose() * wrapped_deltas).transpose();
    for (int j = 0; j < M; ++j) {
      z_mean[j] = wrap_angle(z_pts(0, j) + delta_mean[j]);
      residual[j] = wrap_angle(observed[j] - z_mean[j]);
    }
    for (int k = 0; k < kSigmaCount; ++k) {
      for (int j = 0; j < M; ++j) {
        dz_pts(k, j) = wrap_angle(z_pts(k, j) - z_mean[j]);
      }
    }
  } else {
    z_mean = (w_m.transpose() * z_pts).transpose();
    dz_pts = z_pts.rowwise() - z_mean.transpose();
    residual = observed - z_mean;
  }

  const Eigen::Matrix<double, M, M> s_r = sensor_noise_sqrt<M>(sensor);
  Eigen::Matrix<double, M, M> innovation;
  Eigen::Matrix<double, kDim, M> gain;
  correct_srukf<M>(q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual, innovation,
                   gain);

  if (output != nullptr) {
    Eigen::Matrix<double, M, 1> post_pred;
    Eigen::Matrix<double, M, 1> post_res;
    if (direction_measurement_fixed<M>(sensor, mount_rot, q, post_pred)) {
      post_res = observed - post_pred;
      if (sensor.code == 4 || sensor.code == 5) {
        for (int j = 0; j < M; ++j) {
          post_res[j] = wrap_angle(post_res[j]);
        }
      }
    } else {
      post_res.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
    record_measurement_diagnostics<M>(*output, row, residual, innovation,
                                      post_res, gain);
  }
}

void step_impl(Q &q, Eigen::Vector3d &w, Matrix6 &p,
               const Eigen::Matrix3d &acceleration_psd, double dt,
               const Eigen::VectorXd &readings,
               const std::vector<Sensor> &sensors, double alpha, double beta,
               double kappa, std::map<std::string, Eigen::MatrixXd> *output,
               Eigen::Index row) {
  Matrix6 s_mat = cholesky_psd<kDim>(p);
  predict_sqrt(q, w, s_mat, acceleration_psd, dt, alpha, beta, kappa);

  SigmaQuats q_pts;
  SigmaRates w_pts;
  SigmaWeights w_m, w_c;
  SigmaDeviations dx_pts;

  for (const Sensor &sensor : sensors) {
    if (!sensor.enabled || sensor.code < 0 || sensor.code > 5) {
      continue;
    }
    const Eigen::Index obs_width =
        sensor.code == 2                            ? 4
        : (sensor.code == 0 || sensor.code == 1)    ? 3
        : sensor.code == 3                          ? 1
                                                    : 2;
    if (sensor.offset < 0 || sensor.offset + obs_width > readings.size()) {
      throw std::invalid_argument("sensor reading offset is out of range");
    }
    const auto observed_seg = readings.segment(sensor.offset, obs_width);
    if (!observed_seg.allFinite()) {
      continue;
    }

    sigma_points_from_sqrt(q, w, s_mat, alpha, beta, kappa, q_pts, w_pts, w_m,
                           w_c);
    state_deviations(q_pts, w_pts, q, w, dx_pts);

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
      const Eigen::Matrix<double, kSigmaCount, 3> dz_pts =
          z_pts.rowwise() - z_mean.transpose();
      const Eigen::Vector3d residual = observed - z_mean;
      const Eigen::Matrix3d s_r = sensor_noise_sqrt<3>(sensor);

      Eigen::Matrix3d innovation;
      Eigen::Matrix<double, kDim, 3> gain;
      correct_srukf<3>(q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual,
                       innovation, gain);
      if (output != nullptr) {
        const Eigen::Vector3d post_res = observed - (mount * w + sensor.bias);
        record_measurement_diagnostics<3>(*output, row, residual, innovation,
                                          post_res, gain);
      }
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
      Eigen::Matrix<double, kSigmaCount, 3> dz_pts;
      for (int k = 0; k < kSigmaCount; ++k) {
        dz_pts.row(k) = logmap(z_pts[k] * z_mean_conj).transpose();
      }
      const Q obs_q = quat(observed).normalized();
      const Eigen::Vector3d residual = logmap(obs_q * z_mean_conj);
      const Eigen::Matrix3d s_r = sensor_noise_sqrt<3>(sensor);

      Eigen::Matrix3d innovation;
      Eigen::Matrix<double, kDim, 3> gain;
      correct_srukf<3>(q, w, s_mat, dx_pts, dz_pts, w_c, s_r, residual,
                       innovation, gain);
      if (output != nullptr) {
        const Q post_pred = (mount_q * q).normalized();
        const Eigen::Vector3d post_res =
            logmap(obs_q * post_pred.conjugate());
        record_measurement_diagnostics<3>(*output, row, residual, innovation,
                                          post_res, gain);
      }
    } else if (sensor.code == 1) {
      const Eigen::Vector3d observed = observed_seg;
      process_direction_sensor<3>(q, w, s_mat, sensor, observed, q_pts, dx_pts,
                                  w_m, w_c, output, row);
    } else if (sensor.code == 3) {
      const Eigen::Matrix<double, 1, 1> observed = observed_seg;
      process_direction_sensor<1>(q, w, s_mat, sensor, observed, q_pts, dx_pts,
                                  w_m, w_c, output, row);
    } else {
      const Eigen::Vector2d observed = observed_seg;
      process_direction_sensor<2>(q, w, s_mat, sensor, observed, q_pts, dx_pts,
                                  w_m, w_c, output, row);
    }
  }

  const Matrix6 p_out = s_mat * s_mat.transpose();
  p = 0.5 * (p_out + p_out.transpose());
}

} // namespace

AttitudeSequentialSRUKF::AttitudeSequentialSRUKF(
    Eigen::Vector4d quaternion_in, Eigen::Vector3d rate_in,
    Eigen::MatrixXd covariance_in, Eigen::Matrix3d acceleration_psd_in,
    double alpha_in, double beta_in, double kappa_in,
    std::set<std::string> record_in)
    : AttitudeEstimator(std::move(quaternion_in), std::move(rate_in),
                        std::move(covariance_in),
                        std::move(acceleration_psd_in), std::move(record_in)),
      alpha(alpha_in), beta(beta_in), kappa(kappa_in),
      ukf_params(alpha_in, beta_in, kappa_in) {
  if (covariance.rows() != COVARIANCE_SIZE) {
    throw std::invalid_argument("SRUKF covariance must be 6x6");
  }
  validate_ukf_params(alpha, beta, kappa);
}

State AttitudeSequentialSRUKF::predict(const State &state,
                                       const Eigen::Matrix3d &acceleration_psd,
                                       double dt, double alpha, double beta,
                                       double kappa) {
  if (state.covariance.rows() != COVARIANCE_SIZE ||
      state.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("SRUKF covariance must be 6x6");
  }
  Q q = quat(state.quaternion);
  Eigen::Vector3d w = state.rate;
  Matrix6 s_mat = cholesky_psd<kDim>(state.covariance);
  predict_sqrt(q, w, s_mat, acceleration_psd, dt, alpha, beta, kappa);
  Matrix6 p = s_mat * s_mat.transpose();
  p = 0.5 * (p + p.transpose());
  return {vec(q), w, p};
}

State AttitudeSequentialSRUKF::step(const State &state,
                                    const Eigen::Matrix3d &acceleration_psd,
                                    double dt, const Eigen::VectorXd &readings,
                                    const std::vector<Sensor> &sensors,
                                    double alpha, double beta, double kappa) {
  if (state.covariance.rows() != COVARIANCE_SIZE ||
      state.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("SRUKF covariance must be 6x6");
  }
  Q q = quat(state.quaternion);
  Eigen::Vector3d w = state.rate;
  Matrix6 p = state.covariance;
  step_impl(q, w, p, acceleration_psd, dt, readings, sensors, alpha, beta,
            kappa, nullptr, 0);
  return {vec(q), w, p};
}

std::map<std::string, Eigen::MatrixXd>
AttitudeSequentialSRUKF::allocate_samples(Eigen::Index rows,
                                          const std::set<std::string> &fields) {
  return AttitudeMEKF::allocate_samples(rows, fields);
}

void AttitudeSequentialSRUKF::update_sample(
    State &current, const Eigen::Matrix3d &acceleration_psd, double dt,
    const Eigen::VectorXd &truth, const Eigen::VectorXd &readings,
    const std::vector<Sensor> &sensors,
    std::map<std::string, Eigen::MatrixXd> &output, Eigen::Index row,
    double alpha, double beta, double kappa) {
  if (current.covariance.rows() != COVARIANCE_SIZE ||
      current.covariance.cols() != COVARIANCE_SIZE) {
    throw std::invalid_argument("SRUKF covariance must be 6x6");
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

std::map<std::string, Eigen::MatrixXd> AttitudeSequentialSRUKF::sample(
    const AttitudeSequentialSRUKF &estimator, const Eigen::VectorXd &times,
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

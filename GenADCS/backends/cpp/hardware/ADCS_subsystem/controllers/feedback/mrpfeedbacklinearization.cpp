#include "mrpfeedbacklinearization.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::controllers {

namespace {

Eigen::Vector3d quaternion_to_mrp(const Eigen::Vector4d &q_in) {
  double norm = q_in.norm();
  if (norm == 0.0 || !std::isfinite(norm)) {
    throw std::invalid_argument("quaternion must be finite and nonzero");
  }
  Eigen::Vector4d q = q_in / norm;
  if (q[0] < 0.0) {
    q = -q;
  }
  Eigen::Vector3d sigma = q.tail<3>() / (1.0 + q[0]);
  double s2 = sigma.squaredNorm();
  if (s2 > 1.0) {
    sigma = -sigma / s2;
  }
  return sigma;
}

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
  return m;
}

Eigen::Matrix3d mrp_b_matrix(const Eigen::Vector3d &sigma) {
  const double s2 = sigma.squaredNorm();
  return (1.0 - s2) * Eigen::Matrix3d::Identity() + 2.0 * skew(sigma) +
         2.0 * (sigma * sigma.transpose());
}

Eigen::Matrix3d mrp_b_dot_matrix(const Eigen::Vector3d &sigma,
                                 const Eigen::Vector3d &sigma_dot) {
  const double s_dot_s = sigma.dot(sigma_dot);
  return -2.0 * s_dot_s * Eigen::Matrix3d::Identity() + 2.0 * skew(sigma_dot) +
         2.0 * (sigma_dot * sigma.transpose() + sigma * sigma_dot.transpose());
}

} // namespace

Eigen::Vector3d mrp_feedback_linearized_step(
    const Eigen::Vector4d &q, const Eigen::Vector3d &omega,
    const Eigen::Vector4d &q_d, const Eigen::Vector3d &omega_d,
    const Eigen::Vector3d &omega_d_dot, const Eigen::Vector3d &ff_torque,
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
    const Eigen::Matrix3d &total_inertia, const Eigen::Matrix3d &bus_inertia,
    const Eigen::Ref<const Eigen::MatrixXd> &wheel_axes,
    const Eigen::Matrix3d &kp, const Eigen::Matrix3d &kd) {
  const Eigen::Vector3d sigma = -quaternion_to_mrp(q);
  Eigen::Vector3d sigma_d = -quaternion_to_mrp(q_d);

  const double sd2 = sigma_d.squaredNorm();
  if (sd2 > 1e-12) {
    const Eigen::Vector3d sigma_d_shadow = -sigma_d / sd2;
    if ((sigma - sigma_d_shadow).squaredNorm() <
        (sigma - sigma_d).squaredNorm()) {
      sigma_d = sigma_d_shadow;
    }
  }

  const Eigen::Matrix3d b = mrp_b_matrix(sigma);
  const Eigen::Vector3d sigma_dot = 0.25 * (b * omega);
  const Eigen::Matrix3d b_dot = mrp_b_dot_matrix(sigma, sigma_dot);

  const Eigen::Matrix3d b_d = mrp_b_matrix(sigma_d);
  const Eigen::Vector3d sigma_d_dot = 0.25 * (b_d * omega_d);
  const Eigen::Matrix3d b_d_dot = mrp_b_dot_matrix(sigma_d, sigma_d_dot);
  const Eigen::Vector3d sigma_d_ddot =
      0.25 * (b_d_dot * omega_d) + 0.25 * (b_d * omega_d_dot);

  const Eigen::Vector3d e = sigma - sigma_d;
  const Eigen::Vector3d e_dot = sigma_dot - sigma_d_dot;

  const double s2 = sigma.squaredNorm();
  const double denom = (1.0 + s2) * (1.0 + s2);
  const Eigen::Matrix3d b_inv = b.transpose() / denom;
  const Eigen::Vector3d nu =
      4.0 * (b_inv * (-0.25 * (b_dot * omega) + sigma_d_ddot - kd * e_dot -
                      kp * e));

  Eigen::Vector3d angular_momentum = total_inertia * omega;
  for (Eigen::Index w = 0; w < wheel_axes.rows(); ++w) {
    angular_momentum += wheel_axes.row(w).transpose() * wheel_momenta[w];
  }

  return omega.cross(angular_momentum) + bus_inertia * nu + ff_torque;
}

MRPFeedbackLinearized::MRPFeedbackLinearized(double kp, double kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(double kp,
                                             const Eigen::Vector3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(double kp,
                                             const Eigen::Matrix3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Vector3d &kp,
                                             double kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Vector3d &kp,
                                             const Eigen::Vector3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Vector3d &kp,
                                             const Eigen::Matrix3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Matrix3d &kp,
                                             double kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Matrix3d &kp,
                                             const Eigen::Vector3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

MRPFeedbackLinearized::MRPFeedbackLinearized(const Eigen::Matrix3d &kp,
                                             const Eigen::Matrix3d &kd)
    : kp_(to_gain_matrix(kp, "kp", true)),
      kd_(to_gain_matrix(kd, "kd", false)) {}

} // namespace omni::hardware::controllers

#include "linftyallocator.hpp"

#include <cmath>

namespace omni::hardware::controllers {

Eigen::VectorXd allocate_linfinity(
    const Eigen::Vector3d &tau_c,
    const Eigen::Ref<const Eigen::MatrixXd> &a_pinv,
    const Eigen::Ref<const Eigen::MatrixXd> &null_basis,
    const Eigen::Ref<const Eigen::MatrixXd> &null_proj,
    const Eigen::Ref<const Eigen::VectorXd> &u_0,
    const Eigen::Ref<const Eigen::VectorXd> &max_torque) {
  const Eigen::VectorXd u_base = a_pinv * tau_c;
  const Eigen::Index r = null_basis.cols();
  if (r == 0) {
    return saturate_wheels(u_base, max_torque);
  }

  const Eigen::Index n_w = u_base.size();
  Eigen::VectorXd u;
  if (r == 1) {
    const Eigen::VectorXd v = null_basis.col(0);
    Eigen::VectorXd best_u = u_base;
    double best_max = 0.0;
    for (Eigen::Index i = 0; i < n_w; ++i) {
      const double a = std::abs(best_u[i]);
      if (a > best_max) {
        best_max = a;
      }
    }
    for (Eigen::Index i = 0; i < n_w; ++i) {
      for (Eigen::Index j = i + 1; j < n_w; ++j) {
        for (const double sgn : {1.0, -1.0}) {
          const double denom = v[i] - sgn * v[j];
          if (std::abs(denom) > 1e-12) {
            const double z = -(u_base[i] - sgn * u_base[j]) / denom;
            double cand_max = 0.0;
            for (Eigen::Index k = 0; k < n_w; ++k) {
              const double a = std::abs(u_base[k] + z * v[k]);
              if (a > cand_max) {
                cand_max = a;
              }
            }
            if (cand_max < best_max) {
              best_max = cand_max;
              best_u = u_base + z * v;
            }
          }
        }
      }
    }
    u = best_u;
  } else {
    Eigen::VectorXd z = Eigen::VectorXd::Zero(r);
    u = u_base;
    Eigen::VectorXd best_u = u_base;
    double best_max = 0.0;
    for (Eigen::Index i = 0; i < n_w; ++i) {
      const double a = std::abs(best_u[i]);
      if (a > best_max) {
        best_max = a;
      }
    }
    for (int it = 0; it < 40; ++it) {
      Eigen::Index k_max = 0;
      double u_max = std::abs(u[0]);
      for (Eigen::Index i = 1; i < n_w; ++i) {
        const double a = std::abs(u[i]);
        if (a > u_max) {
          u_max = a;
          k_max = i;
        }
      }
      if (u_max < best_max) {
        best_max = u_max;
        best_u = u;
      }
      const double sgn = (u[k_max] >= 0.0) ? 1.0 : -1.0;
      const Eigen::VectorXd g = sgn * null_basis.row(k_max).transpose();
      const double g_norm = g.norm();
      if (g_norm <= 1e-12) {
        break;
      }
      const double step =
          (0.4 * best_max / std::sqrt(static_cast<double>(it + 1))) / g_norm;
      z -= step * g;
      u = u_base + null_basis * z;
    }
    u = best_u;
  }

  Eigen::VectorXd u_rw = u + null_proj * u_0;
  return saturate_wheels(std::move(u_rw), max_torque);
}

} // namespace omni::hardware::controllers

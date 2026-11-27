#include "hoganschaubdesaturation.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::controllers {

Eigen::VectorXd hogan_schaub_u0(
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta, double gain,
    double target_momentum) {
  return gain * (wheel_momenta.array() - target_momentum).matrix();
}

void hogan_schaub_mtq_dump(
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
    const Eigen::Ref<const Eigen::MatrixXd> &wheel_axes, double mtq_gain,
    double target_momentum, const Eigen::Ref<const Eigen::VectorXd> &readings,
    const SensorConfig &sensor_config,
    const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
    const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
    Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  if (mtq_gain <= 0.0 || dipole_commands.size() == 0) {
    dipole_commands.setZero();
    return;
  }
  const auto [ok_mag, b_body] =
      read_first_sensor_body(1, readings, sensor_config);
  if (!ok_mag) {
    dipole_commands.setZero();
    return;
  }
  const double b_sq = b_body.squaredNorm();
  if (b_sq <= 1e-18) {
    dipole_commands.setZero();
    return;
  }
  Eigen::Vector3d h_err_body = Eigen::Vector3d::Zero();
  for (Eigen::Index w = 0; w < wheel_momenta.size(); ++w) {
    h_err_body +=
        wheel_axes.row(w).transpose() * (wheel_momenta[w] - target_momentum);
  }
  const Eigen::Vector3d dipole_body =
      (mtq_gain / b_sq) * h_err_body.cross(b_body);
  allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

HoganSchaubManagement::HoganSchaubManagement(double gain,
                                             double target_momentum,
                                             double mtq_gain)
    : gain_(gain), target_momentum_(target_momentum), mtq_gain_(mtq_gain) {
  if (!std::isfinite(gain_) || gain_ <= 0.0) {
    throw std::invalid_argument(
        "HoganSchaubManagement gain must be finite and positive");
  }
  if (!std::isfinite(target_momentum_)) {
    throw std::invalid_argument("target_momentum must be finite");
  }
  if (!std::isfinite(mtq_gain_) || mtq_gain_ < 0.0) {
    throw std::invalid_argument("mtq_gain must be finite and nonnegative");
  }
}

Eigen::VectorXd HoganSchaubManagement::to_params() const {
  Eigen::VectorXd p(3);
  p << gain_, target_momentum_, mtq_gain_;
  return p;
}

} // namespace omni::hardware::controllers

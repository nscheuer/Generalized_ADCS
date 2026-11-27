#include "bdot.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::controllers {

void bdot_step(double gain, double dt_ctrl,
               const Eigen::Ref<const Eigen::VectorXd> &readings,
               const SensorConfig &sensor_config,
               const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
               const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
               Eigen::Ref<Eigen::VectorXd> ctrl_work,
               Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  const auto [ok_mag, b_body] =
      read_first_sensor_body(1, readings, sensor_config);
  if (!ok_mag) {
    dipole_commands.setZero();
    return;
  }
  if (ctrl_work[3] == 0.0 || dt_ctrl <= 0.0) {
    ctrl_work.head<3>() = b_body;
    ctrl_work[3] = 1.0;
    dipole_commands.setZero();
    return;
  }
  const Eigen::Vector3d b_dot = (b_body - ctrl_work.head<3>()) / dt_ctrl;
  ctrl_work.head<3>() = b_body;
  const Eigen::Vector3d dipole_body = -gain * b_dot;
  allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

BDot::BDot(double gain) : gain_(gain) {
  if (!std::isfinite(gain_) || gain_ <= 0.0) {
    throw std::invalid_argument("BDot gain must be finite and positive");
  }
}

void BDot::validate_sensors(
    const std::vector<std::string> &sensor_types) const {
  const bool has_mag =
      std::find(sensor_types.begin(), sensor_types.end(), "Magnetometer") !=
      sensor_types.end();
  if (!has_mag) {
    throw std::invalid_argument(
        "BDot controller requires a Magnetometer onboard the satellite");
  }
}

Eigen::VectorXd BDot::to_params() const {
  Eigen::VectorXd p(1);
  p[0] = gain_;
  return p;
}

} // namespace omni::hardware::controllers

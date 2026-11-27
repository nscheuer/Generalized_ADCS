#include "bcross.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::controllers {

void bcross_step(double gain,
                 const Eigen::Ref<const Eigen::VectorXd> &readings,
                 const SensorConfig &sensor_config,
                 const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
                 const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
                 Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  const auto [ok_mag, b_body] =
      read_first_sensor_body(1, readings, sensor_config);
  const auto [ok_gyro, w_body] =
      read_first_sensor_body(0, readings, sensor_config);
  if (!(ok_mag && ok_gyro)) {
    dipole_commands.setZero();
    return;
  }
  const double b_sq = b_body.squaredNorm();
  if (b_sq <= 1e-18) {
    dipole_commands.setZero();
    return;
  }
  const Eigen::Vector3d dipole_body = (gain / b_sq) * w_body.cross(b_body);
  allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

BCross::BCross(double gain) : gain_(gain) {
  if (!std::isfinite(gain_) || gain_ <= 0.0) {
    throw std::invalid_argument("BCross gain must be finite and positive");
  }
}

void BCross::validate_sensors(
    const std::vector<std::string> &sensor_types) const {
  const bool has_mag =
      std::find(sensor_types.begin(), sensor_types.end(), "Magnetometer") !=
      sensor_types.end();
  const bool has_gyro =
      std::find(sensor_types.begin(), sensor_types.end(), "Gyroscope") !=
      sensor_types.end();
  if (!has_mag || !has_gyro) {
    throw std::invalid_argument(
        "BCross controller requires both a Magnetometer and a Gyroscope "
        "onboard the satellite");
  }
}

Eigen::VectorXd BCross::to_params() const {
  Eigen::VectorXd p(1);
  p[0] = gain_;
  return p;
}

} // namespace omni::hardware::controllers

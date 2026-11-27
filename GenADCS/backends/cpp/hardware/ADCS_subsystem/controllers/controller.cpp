#include "controller.hpp"

#include "helpers/math/rotations/rotation_types.hpp"

#include <cmath>

namespace omni::hardware::controllers {

std::pair<bool, Eigen::Vector3d>
read_first_sensor_body(int target_code,
                       const Eigen::Ref<const Eigen::VectorXd> &readings,
                       const SensorConfig &sensor_config) {
  for (const auto &entry : sensor_config) {
    if (entry.code == target_code && entry.enabled &&
        entry.offset >= 0 && entry.offset + 3 <= readings.size()) {
      const Eigen::Vector3d raw = readings.segment<3>(entry.offset);
      if (raw.allFinite()) {
        const Eigen::Matrix3d dcm =
            omni::rotations::quaternion_to_dcm(entry.orientation);
        return {true, dcm.transpose() * raw};
      }
    }
  }
  return {false, Eigen::Vector3d::Zero()};
}

void allocate_dipoles(const Eigen::Vector3d &dipole_body,
                      const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
                      const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
                      Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  for (Eigen::Index i = 0; i < dipole_commands.size(); ++i) {
    double cmd = dipole_body.dot(mtq_axes.row(i).transpose());
    const double limit = mtq_max_dipole[i];
    if (cmd > limit) {
      cmd = limit;
    } else if (cmd < -limit) {
      cmd = -limit;
    }
    dipole_commands[i] = cmd;
  }
}

} // namespace omni::hardware::controllers

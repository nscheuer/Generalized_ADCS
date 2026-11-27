#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_CONTROLLER_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_CONTROLLER_HPP

#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>

namespace omni::hardware::controllers {

inline constexpr int CTRL_NONE = 0;
inline constexpr int CTRL_BDOT = 1;
inline constexpr int CTRL_BCROSS = 2;
inline constexpr int CTRL_ATTITUDE = 3;

struct SensorEntry {
  int code = -1; // 0 = gyroscope, 1 = magnetometer
  Eigen::Vector4d orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  bool enabled = true;
  Eigen::Index offset = 0;
};

using SensorConfig = std::vector<SensorEntry>;

std::pair<bool, Eigen::Vector3d>
read_first_sensor_body(int target_code,
                       const Eigen::Ref<const Eigen::VectorXd> &readings,
                       const SensorConfig &sensor_config);

void allocate_dipoles(const Eigen::Vector3d &dipole_body,
                      const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
                      const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
                      Eigen::Ref<Eigen::VectorXd> dipole_commands);

class Controller {
public:
  virtual ~Controller() = default;
  virtual int kind() const = 0;
  virtual void
  validate_sensors(const std::vector<std::string> &sensor_types) const = 0;
  virtual Eigen::VectorXd to_params() const = 0;
};

class NoController final : public Controller {
public:
  int kind() const override { return CTRL_NONE; }
  void
  validate_sensors(const std::vector<std::string> & /*sensor_types*/) const override {}
  Eigen::VectorXd to_params() const override { return Eigen::VectorXd::Zero(1); }
};

} // namespace omni::hardware::controllers

#endif

#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_RUNTIME_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_RUNTIME_HPP

#include "controller.hpp"

#include <Eigen/Core>
#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace omni::hardware::controllers {

inline constexpr int CTRL_PARAM_WIDTH = 7;

class ControllerTable {
public:
  ControllerTable();
  std::size_t register_controller(const std::shared_ptr<Controller> &controller);

  std::vector<int> kinds;
  std::vector<Eigen::Matrix<double, CTRL_PARAM_WIDTH, 1>> params;
  std::vector<Eigen::Matrix3d> kp;
  std::vector<Eigen::Matrix3d> kd;
  std::vector<std::shared_ptr<Controller>> objects;
};

struct ControllerRuntime {
  bool enabled = false;
  std::vector<int> kinds;
  Eigen::MatrixXd params;
  std::vector<Eigen::Matrix3d> kp;
  std::vector<Eigen::Matrix3d> kd;
  Eigen::MatrixXd mtq_axes;
  Eigen::VectorXd mtq_max_dipole;
  Eigen::MatrixXd wheel_axes;
  Eigen::VectorXd wheel_max_torque;
  Eigen::MatrixXd a_pinv;
  Eigen::MatrixXd null_basis;
  Eigen::MatrixXd null_proj;
  Eigen::Matrix3d total_inertia = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d bus_inertia = Eigen::Matrix3d::Identity();
};

ControllerRuntime prepare_controllers(
    const std::vector<std::shared_ptr<Controller>> &controllers,
    const std::vector<std::string> &sensor_types,
    const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
    const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
    ControllerTable *table = nullptr,
    const Eigen::MatrixXd *wheel_axes = nullptr,
    const Eigen::VectorXd *wheel_max_torque = nullptr,
    const Eigen::Matrix3d *total_inertia = nullptr,
    const Eigen::Matrix3d *bus_inertia = nullptr);

Eigen::VectorXd init_controller_work();

void sync_actuator_enable(const std::vector<bool> &rw_enabled,
                          const std::vector<bool> &mtq_enabled,
                          const Eigen::Ref<const Eigen::VectorXd> &base_wheel_max,
                          const Eigen::Ref<const Eigen::VectorXd> &base_mtq_max,
                          ControllerRuntime &ctrl_rt,
                          Eigen::Ref<Eigen::VectorXd> wheel_commands,
                          Eigen::Ref<Eigen::VectorXd> dipole_commands);

std::tuple<Eigen::Vector3d, Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector3d>
step_controller(std::size_t ctrl_idx, double dt_ctrl, const Eigen::Vector4d &q,
                const Eigen::Vector3d &omega, const Eigen::Vector4d &q_goal,
                const Eigen::Vector3d &w_goal,
                const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
                const Eigen::Ref<const Eigen::VectorXd> &readings,
                const SensorConfig &sensor_config,
                const ControllerRuntime &ctrl_rt,
                Eigen::Ref<Eigen::VectorXd> ctrl_work,
                Eigen::Ref<Eigen::VectorXd> wheel_commands,
                Eigen::Ref<Eigen::VectorXd> dipole_commands);

} // namespace omni::hardware::controllers

#endif

#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/bdot.hpp"
#include "hardware/ADCS_subsystem/controllers/runtime.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace omni::hardware::controllers;

namespace {
Eigen::Vector4d quat_mul(const Eigen::Vector4d &a, const Eigen::Vector4d &b) {
  return Eigen::Vector4d(
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}
} // namespace

int main() {
  bool threw_gain = false;
  try {
    BDot bad(0.0);
  } catch (const std::invalid_argument &) {
    threw_gain = true;
  }
  CHECK(threw_gain);

  auto bdot = std::make_shared<BDot>(5e4);
  bool threw_sensor = false;
  try {
    bdot->validate_sensors({"Gyroscope"});
  } catch (const std::invalid_argument &) {
    threw_sensor = true;
  }
  CHECK(threw_sensor);
  bdot->validate_sensors({"Magnetometer"});

  Eigen::MatrixXd mtq_axes = Eigen::Matrix3d::Identity();
  Eigen::VectorXd mtq_max = Eigen::VectorXd::Ones(3);
  ControllerTable table;
  const std::size_t idx = table.register_controller(bdot);
  const ControllerRuntime rt =
      prepare_controllers({}, {"Magnetometer"}, mtq_axes, mtq_max, &table);

  SensorConfig sc;
  sc.push_back(SensorEntry{1, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), true, 0});

  Eigen::VectorXd ctrl_work = init_controller_work();
  Eigen::VectorXd wheel_cmd = Eigen::VectorXd::Zero(0);
  Eigen::VectorXd dipole_cmd = Eigen::VectorXd::Zero(3);

  Eigen::Vector4d q(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d omega(0.05, -0.04, 0.03);
  const double initial_norm = omega.norm();
  const Eigen::Matrix3d inertia = Eigen::Vector3d(0.1, 0.1, 0.1).asDiagonal();
  const Eigen::Matrix3d inv_inertia = inertia.inverse();
  const Eigen::Vector3d b_inertial(2.0e-5, -1.5e-5, 3.0e-5);
  const double dt = 0.25;

  for (int step = 0; step < 480; ++step) {
    const Eigen::Matrix3d r_bi =
        Eigen::Quaterniond(q[0], q[1], q[2], q[3]).toRotationMatrix().transpose();
    const Eigen::Vector3d b_body = r_bi * b_inertial;
    step_controller(idx, dt, q, omega, Eigen::Vector4d::Zero(),
                    Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(0), b_body,
                    sc, rt, ctrl_work, wheel_cmd, dipole_cmd);
    const Eigen::Vector3d torque = dipole_cmd.head<3>().cross(b_body);
    omega += dt * (inv_inertia * (torque - omega.cross(inertia * omega)));
    const Eigen::Vector4d w_q(0.0, omega[0], omega[1], omega[2]);
    q += 0.5 * dt * quat_mul(q, w_q);
    q.normalize();
  }

  CHECK(omega.norm() < initial_norm);
  return 0;
}

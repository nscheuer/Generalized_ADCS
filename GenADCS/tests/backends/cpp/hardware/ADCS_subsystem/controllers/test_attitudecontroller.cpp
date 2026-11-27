#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/allocator/l2allocator.hpp"
#include "hardware/ADCS_subsystem/controllers/allocator/linftyallocator.hpp"
#include "hardware/ADCS_subsystem/controllers/attitude_controller.hpp"
#include "hardware/ADCS_subsystem/controllers/desaturation/hoganschaubdesaturation.hpp"
#include "hardware/ADCS_subsystem/controllers/feedback/mrpfeedbacklinearization.hpp"
#include "hardware/ADCS_subsystem/controllers/feedforward/passthrough.hpp"
#include "hardware/ADCS_subsystem/controllers/runtime.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <memory>

using namespace omni::hardware::controllers;

namespace {

Eigen::Vector4d quat_mul(const Eigen::Vector4d &a, const Eigen::Vector4d &b) {
  return Eigen::Vector4d(
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

void rk4_step(Eigen::Vector4d &q, Eigen::Vector3d &omega,
              Eigen::VectorXd &wheel_h, const Eigen::VectorXd &wheel_cmd,
              const Eigen::MatrixXd &wheel_axes,
              const Eigen::Matrix3d &total_inertia,
              const Eigen::Matrix3d &bus_inertia, double dt) {
  const Eigen::Matrix3d inv_bus = bus_inertia.inverse();
  const Eigen::Vector3d tau_rw = wheel_axes.transpose() * wheel_cmd;

  auto rhs = [&](const Eigen::Vector4d &qs, const Eigen::Vector3d &ws,
                 const Eigen::VectorXd &hs) {
    const Eigen::Vector4d w_q(0.0, -ws[0], -ws[1], -ws[2]);
    const Eigen::Vector4d qdot = 0.5 * quat_mul(w_q, qs);
    const Eigen::Vector3d h_tot =
        total_inertia * ws + wheel_axes.transpose() * hs;
    const Eigen::Vector3d wdot = inv_bus * (tau_rw - ws.cross(h_tot));
    const Eigen::VectorXd hdot = -wheel_cmd;
    return std::tuple<Eigen::Vector4d, Eigen::Vector3d, Eigen::VectorXd>(
        qdot, wdot, hdot);
  };

  auto [k1q, k1w, k1h] = rhs(q, omega, wheel_h);
  auto [k2q, k2w, k2h] =
      rhs(q + 0.5 * dt * k1q, omega + 0.5 * dt * k1w, wheel_h + 0.5 * dt * k1h);
  auto [k3q, k3w, k3h] =
      rhs(q + 0.5 * dt * k2q, omega + 0.5 * dt * k2w, wheel_h + 0.5 * dt * k2h);
  auto [k4q, k4w, k4h] = rhs(q + dt * k3q, omega + dt * k3w, wheel_h + dt * k3h);

  q += (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
  q.normalize();
  omega += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
  wheel_h += (dt / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h);
}

void test_tracks_quaternion_goal() {
  const Eigen::Matrix3d inertia =
      Eigen::Vector3d(0.15, 0.12, 0.08).asDiagonal();
  const Eigen::MatrixXd wheel_axes = Eigen::Matrix3d::Identity();
  const Eigen::VectorXd wheel_max = Eigen::VectorXd::Constant(3, 0.2);

  auto ctrl = std::make_shared<AttitudeController>(
      std::make_shared<MRPFeedbackLinearized>(0.5, 1.4),
      std::make_shared<Passthrough>(), std::make_shared<L2Allocator>(),
      std::make_shared<NoDesaturation>());

  ControllerTable table;
  const std::size_t idx = table.register_controller(ctrl);
  const ControllerRuntime rt = prepare_controllers(
      {}, {}, Eigen::MatrixXd::Zero(0, 3), Eigen::VectorXd::Zero(0), &table,
      &wheel_axes, &wheel_max, &inertia, &inertia);

  const double pi = std::acos(-1.0);
  Eigen::Vector4d q(std::cos(pi / 6.0), std::sin(pi / 6.0), 0.0, 0.0);
  Eigen::Vector3d omega(0.05, -0.04, 0.03);
  Eigen::VectorXd wheel_h = Eigen::VectorXd::Zero(3);
  const Eigen::Vector4d target_q(1.0, 0.0, 0.0, 0.0);
  const Eigen::Vector3d target_w = Eigen::Vector3d::Zero();

  SensorConfig sc;
  Eigen::VectorXd ctrl_work = init_controller_work();
  Eigen::VectorXd wheel_cmd = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd dipole_cmd = Eigen::VectorXd::Zero(0);
  const double dt = 0.05;

  for (int step = 0; step < 400; ++step) {
    step_controller(idx, dt, q, omega, target_q, target_w, wheel_h,
                    Eigen::VectorXd::Zero(0), sc, rt, ctrl_work, wheel_cmd,
                    dipole_cmd);
    rk4_step(q, omega, wheel_h, wheel_cmd, wheel_axes, inertia, inertia, dt);
  }

  CHECK(std::abs(q.dot(target_q)) > 0.999);
  CHECK(omega.norm() < 1e-3);
}

void test_4_wheels_linfinity_and_hogan_schaub() {
  const double s3 = 1.0 / std::sqrt(3.0);
  Eigen::MatrixXd wheel_axes(4, 3);
  wheel_axes << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, s3, s3, s3;
  const Eigen::VectorXd wheel_max = Eigen::VectorXd::Constant(4, 0.2);
  const Eigen::Matrix3d inertia =
      Eigen::Vector3d(0.15, 0.12, 0.08).asDiagonal();

  auto ctrl = std::make_shared<AttitudeController>(
      std::make_shared<MRPFeedbackLinearized>(0.5, 1.4),
      std::make_shared<Passthrough>(), std::make_shared<LInfinityAllocator>(),
      std::make_shared<HoganSchaubManagement>(0.5, 0.0, 0.0));

  ControllerTable table;
  const std::size_t idx = table.register_controller(ctrl);
  const ControllerRuntime rt = prepare_controllers(
      {}, {}, Eigen::MatrixXd::Zero(0, 3), Eigen::VectorXd::Zero(0), &table,
      &wheel_axes, &wheel_max, &inertia, &inertia);

  const Eigen::VectorXd null_vec = rt.null_basis.col(0);
  Eigen::VectorXd wheel_h = 0.4 * null_vec;
  const double h0_norm = wheel_h.norm();
  Eigen::Vector4d q(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();

  SensorConfig sc;
  Eigen::VectorXd ctrl_work = init_controller_work();
  Eigen::VectorXd wheel_cmd = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd dipole_cmd = Eigen::VectorXd::Zero(0);
  const double dt = 0.05;

  for (int step = 0; step < 300; ++step) {
    step_controller(idx, dt, q, omega, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                    Eigen::Vector3d::Zero(), wheel_h, Eigen::VectorXd::Zero(0),
                    sc, rt, ctrl_work, wheel_cmd, dipole_cmd);
    rk4_step(q, omega, wheel_h, wheel_cmd, wheel_axes, inertia, inertia, dt);
  }

  CHECK(wheel_h.norm() < 0.05 * h0_norm);
}

} // namespace

int main() {
  test_tracks_quaternion_goal();
  test_4_wheels_linfinity_and_hogan_schaub();
  return 0;
}

#include "runtime.hpp"

#include "attitude_controller.hpp"
#include "bcross.hpp"
#include "bdot.hpp"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omni::hardware::controllers {

namespace {

void compute_allocation_matrices(
    const Eigen::MatrixXd &a_mat, Eigen::MatrixXd &a_pinv,
    Eigen::MatrixXd &null_proj, Eigen::MatrixXd &null_basis) {
  const Eigen::Index n_w = a_mat.cols();
  if (n_w == 0) {
    a_pinv = Eigen::MatrixXd::Zero(0, 3);
    null_proj = Eigen::MatrixXd::Zero(0, 0);
    null_basis = Eigen::MatrixXd::Zero(0, 0);
    return;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      a_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::VectorXd &s = svd.singularValues();
  const double tol_pinv =
      (s.size() > 0)
          ? 1e-15 * static_cast<double>(std::max(a_mat.rows(), a_mat.cols())) *
                s[0]
          : 0.0;
  Eigen::MatrixXd s_pinv = Eigen::MatrixXd::Zero(n_w, 3);
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    if (s[i] > tol_pinv) {
      s_pinv(i, i) = 1.0 / s[i];
    }
    if (s[i] > 1e-12) {
      ++rank;
    }
  }
  a_pinv = svd.matrixV() * s_pinv * svd.matrixU().transpose();
  null_proj = Eigen::MatrixXd::Identity(n_w, n_w) - a_pinv * a_mat;
  const Eigen::Index null_cols = n_w - rank;
  null_basis = svd.matrixV().rightCols(null_cols);
}

} // namespace

ControllerTable::ControllerTable() {
  kinds.push_back(CTRL_NONE);
  params.push_back(Eigen::Matrix<double, CTRL_PARAM_WIDTH, 1>::Zero());
  kp.push_back(Eigen::Matrix3d::Zero());
  kd.push_back(Eigen::Matrix3d::Zero());
  objects.push_back(std::make_shared<NoController>());
}

std::size_t ControllerTable::register_controller(
    const std::shared_ptr<Controller> &controller) {
  if (!controller || controller->kind() == CTRL_NONE) {
    return 0;
  }
  Eigen::Matrix<double, CTRL_PARAM_WIDTH, 1> p =
      Eigen::Matrix<double, CTRL_PARAM_WIDTH, 1>::Zero();
  const Eigen::VectorXd raw = controller->to_params();
  const Eigen::Index copy_len =
      std::min<Eigen::Index>(raw.size(), CTRL_PARAM_WIDTH);
  p.head(copy_len) = raw.head(copy_len);

  Eigen::Matrix3d kp_mat = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d kd_mat = Eigen::Matrix3d::Zero();
  if (const auto *att =
          dynamic_cast<const AttitudeController *>(controller.get())) {
    std::tie(kp_mat, kd_mat) = att->feedback()->to_matrices();
  }

  const std::size_t idx = kinds.size();
  kinds.push_back(controller->kind());
  params.push_back(p);
  kp.push_back(kp_mat);
  kd.push_back(kd_mat);
  objects.push_back(controller);
  return idx;
}

ControllerRuntime prepare_controllers(
    const std::vector<std::shared_ptr<Controller>> &controllers,
    const std::vector<std::string> &sensor_types,
    const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
    const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
    ControllerTable *table, const Eigen::MatrixXd *wheel_axes,
    const Eigen::VectorXd *wheel_max_torque,
    const Eigen::Matrix3d *total_inertia, const Eigen::Matrix3d *bus_inertia) {
  ControllerTable local_table;
  ControllerTable &tbl = (table != nullptr) ? *table : local_table;

  for (const auto &ctrl : controllers) {
    tbl.register_controller(ctrl);
  }
  for (const auto &ctrl : tbl.objects) {
    ctrl->validate_sensors(sensor_types);
  }

  ControllerRuntime rt;
  rt.enabled = std::any_of(tbl.kinds.begin(), tbl.kinds.end(),
                           [](int k) { return k != CTRL_NONE; });
  rt.kinds = tbl.kinds;
  rt.params.resize(static_cast<Eigen::Index>(tbl.params.size()),
                   CTRL_PARAM_WIDTH);
  for (std::size_t i = 0; i < tbl.params.size(); ++i) {
    rt.params.row(static_cast<Eigen::Index>(i)) = tbl.params[i].transpose();
  }
  rt.kp = tbl.kp;
  rt.kd = tbl.kd;
  rt.mtq_axes = mtq_axes;
  rt.mtq_max_dipole = mtq_max_dipole;

  if (wheel_axes != nullptr) {
    rt.wheel_axes = *wheel_axes;
  } else {
    rt.wheel_axes = Eigen::MatrixXd::Zero(0, 3);
  }
  if (wheel_max_torque != nullptr) {
    rt.wheel_max_torque = *wheel_max_torque;
  } else {
    rt.wheel_max_torque = Eigen::VectorXd::Zero(0);
  }

  compute_allocation_matrices(rt.wheel_axes.transpose(), rt.a_pinv,
                              rt.null_proj, rt.null_basis);

  rt.total_inertia = (total_inertia != nullptr) ? *total_inertia
                                                : Eigen::Matrix3d::Identity();
  rt.bus_inertia =
      (bus_inertia != nullptr) ? *bus_inertia : rt.total_inertia;
  return rt;
}

Eigen::VectorXd init_controller_work() { return Eigen::VectorXd::Zero(4); }

void sync_actuator_enable(const std::vector<bool> &rw_enabled,
                          const std::vector<bool> &mtq_enabled,
                          const Eigen::Ref<const Eigen::VectorXd> &base_wheel_max,
                          const Eigen::Ref<const Eigen::VectorXd> &base_mtq_max,
                          ControllerRuntime &ctrl_rt,
                          Eigen::Ref<Eigen::VectorXd> wheel_commands,
                          Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  for (std::size_t i = 0; i < mtq_enabled.size(); ++i) {
    const auto idx = static_cast<Eigen::Index>(i);
    if (mtq_enabled[i]) {
      ctrl_rt.mtq_max_dipole[idx] = base_mtq_max[idx];
    } else {
      ctrl_rt.mtq_max_dipole[idx] = 0.0;
      dipole_commands[idx] = 0.0;
    }
  }

  const std::size_t n_w = rw_enabled.size();
  if (n_w == 0) {
    return;
  }
  std::size_t active_wheels = 0;
  for (std::size_t i = 0; i < n_w; ++i) {
    const auto idx = static_cast<Eigen::Index>(i);
    if (rw_enabled[i]) {
      ctrl_rt.wheel_max_torque[idx] = base_wheel_max[idx];
      ++active_wheels;
    } else {
      ctrl_rt.wheel_max_torque[idx] = 0.0;
      wheel_commands[idx] = 0.0;
    }
  }

  if (active_wheels == 0) {
    ctrl_rt.a_pinv.setZero();
    ctrl_rt.null_proj.setZero();
    if (ctrl_rt.null_basis.size() > 0) {
      ctrl_rt.null_basis.setZero();
    }
    return;
  }

  Eigen::MatrixXd a_mat =
      Eigen::MatrixXd::Zero(3, static_cast<Eigen::Index>(n_w));
  for (std::size_t i = 0; i < n_w; ++i) {
    if (rw_enabled[i]) {
      const auto idx = static_cast<Eigen::Index>(i);
      a_mat.col(idx) = ctrl_rt.wheel_axes.row(idx).transpose();
    }
  }

  Eigen::MatrixXd a_pinv;
  Eigen::MatrixXd null_proj;
  Eigen::MatrixXd full_null_basis;
  compute_allocation_matrices(a_mat, a_pinv, null_proj, full_null_basis);
  ctrl_rt.a_pinv = a_pinv;
  ctrl_rt.null_proj = null_proj;
  if (ctrl_rt.null_basis.cols() > 0) {
    ctrl_rt.null_basis.setZero();
    const Eigen::Index cols = ctrl_rt.null_basis.cols();
    const Eigen::Index avail = full_null_basis.cols();
    const Eigen::Index take = std::min(cols, avail);
    for (Eigen::Index c = 0; c < take; ++c) {
      for (std::size_t r = 0; r < n_w; ++r) {
        if (rw_enabled[r]) {
          ctrl_rt.null_basis(static_cast<Eigen::Index>(r), c) =
              full_null_basis(static_cast<Eigen::Index>(r), c);
        }
      }
    }
  }
}

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
                Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Vector4d nan_q = Eigen::Vector4d::Constant(nan_val);
  const Eigen::Vector3d nan_3 = Eigen::Vector3d::Constant(nan_val);

  if (ctrl_idx >= ctrl_rt.kinds.size()) {
    return {nan_3, nan_q, nan_3, nan_3};
  }
  const int kind = ctrl_rt.kinds[ctrl_idx];
  const auto idx = static_cast<Eigen::Index>(ctrl_idx);

  if (kind == CTRL_NONE) {
    dipole_commands.setZero();
    wheel_commands.setZero();
    return {nan_3, nan_q, nan_3, nan_3};
  }

  if (kind == CTRL_BDOT) {
    wheel_commands.setZero();
    if (dipole_commands.size() > 0) {
      bdot_step(ctrl_rt.params(idx, 0), dt_ctrl, readings, sensor_config,
                ctrl_rt.mtq_axes, ctrl_rt.mtq_max_dipole, ctrl_work,
                dipole_commands);
    }
    return {nan_3, nan_q, nan_3, nan_3};
  }

  if (kind == CTRL_BCROSS) {
    wheel_commands.setZero();
    if (dipole_commands.size() > 0) {
      bcross_step(ctrl_rt.params(idx, 0), readings, sensor_config,
                  ctrl_rt.mtq_axes, ctrl_rt.mtq_max_dipole, dipole_commands);
    }
    return {nan_3, nan_q, nan_3, nan_3};
  }

  if (kind == CTRL_ATTITUDE) {
    return attitude_controller_step(
        q, omega, q_goal, w_goal, wheel_momenta, ctrl_rt.total_inertia,
        ctrl_rt.bus_inertia, ctrl_rt.wheel_axes, ctrl_rt.wheel_max_torque,
        ctrl_rt.a_pinv, ctrl_rt.null_basis, ctrl_rt.null_proj,
        static_cast<int>(ctrl_rt.params(idx, 0)),
        static_cast<int>(ctrl_rt.params(idx, 1)),
        static_cast<int>(ctrl_rt.params(idx, 2)),
        static_cast<int>(ctrl_rt.params(idx, 3)), ctrl_rt.kp[ctrl_idx],
        ctrl_rt.kd[ctrl_idx], ctrl_rt.params(idx, 4), ctrl_rt.params(idx, 5),
        ctrl_rt.params(idx, 6), readings, sensor_config, ctrl_rt.mtq_axes,
        ctrl_rt.mtq_max_dipole, wheel_commands, dipole_commands);
  }

  return {nan_3, nan_q, nan_3, nan_3};
}

} // namespace omni::hardware::controllers

#include "control_loop.hpp"

#include "../helpers/math/time/conversions.hpp"
#include <cmath>
#include <limits>

namespace omni::propagation {

namespace {

void apply_obc_state_masks(
    int state_id, const omni::hardware::obc::OBCRuntime &obc_rt,
    std::vector<bool> &sensor_enabled, std::vector<bool> &rw_enabled,
    std::vector<bool> &mtq_enabled) {
  if (state_id < 0 || static_cast<std::size_t>(state_id) >=
                          obc_rt.state_sensor_mask.size()) {
    return;
  }
  const auto idx = static_cast<std::size_t>(state_id);
  omni::hardware::apply_hw_mask(obc_rt.state_sensor_mask[idx], sensor_enabled);
  omni::hardware::apply_hw_mask(obc_rt.state_rw_mask[idx], rw_enabled);
  omni::hardware::apply_hw_mask(obc_rt.state_mtq_mask[idx], mtq_enabled);
}

void apply_obc_tc_masks(
    int tc_idx, const omni::hardware::obc::OBCRuntime &obc_rt,
    std::vector<bool> &sensor_enabled, std::vector<bool> &rw_enabled,
    std::vector<bool> &mtq_enabled) {
  if (tc_idx < 0 || static_cast<std::size_t>(tc_idx) >=
                        obc_rt.tc_trans_sensor_mask.size()) {
    return;
  }
  const auto idx = static_cast<std::size_t>(tc_idx);
  omni::hardware::apply_hw_mask(obc_rt.tc_trans_sensor_mask[idx],
                                sensor_enabled);
  omni::hardware::apply_hw_mask(obc_rt.tc_trans_rw_mask[idx], rw_enabled);
  omni::hardware::apply_hw_mask(obc_rt.tc_trans_mtq_mask[idx], mtq_enabled);
}

void apply_adcs_mode_masks(
    int mode_id, const omni::hardware::ADCSSMRuntime &adcs_sm_rt,
    std::vector<bool> &sensor_enabled, std::vector<bool> &rw_enabled,
    std::vector<bool> &mtq_enabled) {
  if (!adcs_sm_rt.enabled || mode_id < 0 ||
      static_cast<std::size_t>(mode_id) >= adcs_sm_rt.mode_sensor_mask.size()) {
    return;
  }
  const auto idx = static_cast<std::size_t>(mode_id);
  omni::hardware::apply_hw_mask(adcs_sm_rt.mode_sensor_mask[idx],
                                sensor_enabled);
  omni::hardware::apply_hw_mask(adcs_sm_rt.mode_rw_mask[idx], rw_enabled);
  omni::hardware::apply_hw_mask(adcs_sm_rt.mode_mtq_mask[idx], mtq_enabled);
}

} // namespace

void ControlLoopContext::allocate(Eigen::Index rows, Eigen::Index n_wheels,
                                  Eigen::Index n_mtqs,
                                  const std::set<std::string> &fields) {
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  last_ff_torque = Eigen::Vector3d::Constant(nan_v);
  last_ff_q = Eigen::Vector4d::Constant(nan_v);
  last_ff_w = Eigen::Vector3d::Constant(nan_v);
  last_ref_torque = Eigen::Vector3d::Constant(nan_v);

  record_obc_mode = fields.count("obc_mode") > 0;
  record_adcs_mode = fields.count("adcs_mode") > 0;
  record_adcs_goal = fields.count("adcs_goal") > 0;

  ctrl_log_flags =
      ((fields.count("goal_attitude") || fields.count("goal_rate")) ? 1 : 0) |
      (fields.count("feedforward_torque") ? 2 : 0) |
      (fields.count("feedforward_attitude") ? 4 : 0) |
      (fields.count("feedforward_rate") ? 8 : 0) |
      (fields.count("reference_torque") ? 16 : 0) |
      (fields.count("commanded_wheel_torque") ? 32 : 0) |
      (fields.count("commanded_magnetic_dipole") ? 64 : 0);

  if (record_obc_mode)
    output["obc_mode"] = Eigen::MatrixXd::Zero(rows, 1);
  if (record_adcs_mode)
    output["adcs_mode"] = Eigen::MatrixXd::Zero(rows, 1);
  if (record_adcs_goal)
    output["adcs_goal"] = Eigen::MatrixXd::Zero(rows, 1);
  if (fields.count("goal_attitude"))
    output["goal_attitude"] = Eigen::MatrixXd::Constant(rows, 4, nan_v);
  if (fields.count("goal_rate"))
    output["goal_rate"] = Eigen::MatrixXd::Constant(rows, 3, nan_v);
  if (fields.count("feedforward_torque"))
    output["feedforward_torque"] = Eigen::MatrixXd::Constant(rows, 3, nan_v);
  if (fields.count("feedforward_attitude"))
    output["feedforward_attitude"] = Eigen::MatrixXd::Constant(rows, 4, nan_v);
  if (fields.count("feedforward_rate"))
    output["feedforward_rate"] = Eigen::MatrixXd::Constant(rows, 3, nan_v);
  if (fields.count("reference_torque"))
    output["reference_torque"] = Eigen::MatrixXd::Constant(rows, 3, nan_v);
  if (fields.count("commanded_wheel_torque"))
    output["commanded_wheel_torque"] = Eigen::MatrixXd::Zero(rows, n_wheels);
  if (fields.count("commanded_magnetic_dipole"))
    output["commanded_magnetic_dipole"] = Eigen::MatrixXd::Zero(rows, n_mtqs);
}

void ControlLoopContext::on_init(const Vector &y, double initial_jd, double dt,
                                 int sensor_every, Runtime &runtime,
                                 SimulationSampler &sampler) {
  active_obc_mode = static_cast<int>(obc_work[0]);
  active_adcs_mode = adcs_work[0];
  active_adcs_goal = adcs_work[1];
  prev_ctrl_idx = -1;
  pending_cmd_mode = -1;
  pending_cmd_goal = -1;

  bool hw_dirty = false;
  if (obc_every > 0) {
    const int init_state = static_cast<int>(obc_work[0]);
    pending_cmd_mode = obc_rt.state_adcs_mode[static_cast<std::size_t>(init_state)];
    pending_cmd_goal = obc_rt.state_adcs_goal[static_cast<std::size_t>(init_state)];
    apply_obc_state_masks(init_state, obc_rt, sensor_enabled, rw_enabled,
                          mtq_enabled);
    const auto [st, cm, cg, entered_state, fired_tc_idx] =
        omni::hardware::obc::step_obc_core(y, 0.0, initial_jd, dt * obc_every,
                                           obc_rt, obc_work);
    active_obc_mode = st;
    if (entered_state >= 0) {
      apply_obc_state_masks(entered_state, obc_rt, sensor_enabled, rw_enabled,
                            mtq_enabled);
    }
    if (fired_tc_idx >= 0) {
      apply_obc_tc_masks(fired_tc_idx, obc_rt, sensor_enabled, rw_enabled,
                         mtq_enabled);
    }
    if (cm >= 0)
      pending_cmd_mode = cm;
    if (cg >= 0)
      pending_cmd_goal = cg;
    hw_dirty = true;
  }

  if (adcs_sm_every > 0) {
    const auto [am, ag] = omni::hardware::step_adcs_sm(
        0.0, adcs_sm_rt, adcs_work, pending_cmd_mode, pending_cmd_goal);
    active_adcs_mode = am;
    active_adcs_goal = ag;
    pending_cmd_mode = -1;
    pending_cmd_goal = -1;
    apply_adcs_mode_masks(active_adcs_mode, adcs_sm_rt, sensor_enabled,
                          rw_enabled, mtq_enabled);
    hw_dirty = true;
  }

  if (hw_dirty) {
    sampler.set_sensor_enabled(sensor_enabled);
    for (std::size_t i = 0; i < ctrl_sensor_config.size() && i < sensor_enabled.size(); ++i) {
      ctrl_sensor_config[i].enabled = sensor_enabled[i];
    }
    omni::hardware::controllers::sync_actuator_enable(
        rw_enabled, mtq_enabled, base_wheel_max, base_mtq_max, ctrl_rt,
        runtime.wheel_commands, runtime.dipole_commands);
  }

  if (sensor_every > 0) {
    sampler.sample(y, 0.0, 0);
    if (ctrl_rt.enabled) {
      const auto [gq, gw] = omni::hardware::evaluate_goal(
          active_adcs_goal, adcs_sm_rt, y, initial_jd, goal_runtime);
      const std::size_t ctrl_idx =
          adcs_sm_rt.enabled
              ? static_cast<std::size_t>(
                    adcs_sm_rt.mode_default_controller[static_cast<std::size_t>(
                        active_adcs_mode)])
              : default_ctrl_idx;
      if (static_cast<int>(ctrl_idx) != prev_ctrl_idx) {
        ctrl_work.setZero();
        prev_ctrl_idx = static_cast<int>(ctrl_idx);
      }
      const Eigen::Vector4d q_est =
          sampler.has_estimator() ? sampler.estimate().quaternion
                                  : Eigen::Vector4d(y.head<4>());
      const Eigen::Vector3d w_est =
          sampler.has_estimator() ? sampler.estimate().rate
                                  : Eigen::Vector3d(y.segment<3>(4));
      const Eigen::VectorXd h_wheels =
          (y.size() > 13) ? Eigen::VectorXd(y.tail(y.size() - 13))
                          : Eigen::VectorXd::Zero(0);
      std::tie(last_ff_torque, last_ff_q, last_ff_w, last_ref_torque) =
          omni::hardware::controllers::step_controller(
              ctrl_idx, dt * sensor_every, q_est, w_est, gq, gw, h_wheels,
              sampler.readings(), ctrl_sensor_config, ctrl_rt, ctrl_work,
              runtime.wheel_commands, runtime.dipole_commands);
    }
  }
}

void ControlLoopContext::on_step_pre_record(
    const Vector &y, std::size_t completed, std::size_t full_steps, double dt,
    double initial_jd, int sensor_every, bool sensor_tick, Runtime &runtime,
    SimulationSampler &sampler) {
  const double elapsed_now = static_cast<double>(completed) * dt;
  const double epoch_now = initial_jd + elapsed_now / 86400.0;
  bool hw_dirty = false;

  if (obc_every > 0 && completed <= full_steps &&
      completed % static_cast<std::size_t>(obc_every) == 0) {
    const auto [st, cm, cg, entered_state, fired_tc_idx] =
        omni::hardware::obc::step_obc_core(
            y, elapsed_now, epoch_now, dt * obc_every, obc_rt, obc_work);
    active_obc_mode = st;
    if (entered_state >= 0) {
      apply_obc_state_masks(entered_state, obc_rt, sensor_enabled, rw_enabled,
                            mtq_enabled);
      hw_dirty = true;
    }
    if (fired_tc_idx >= 0) {
      apply_obc_tc_masks(fired_tc_idx, obc_rt, sensor_enabled, rw_enabled,
                         mtq_enabled);
      hw_dirty = true;
    }
    if (cm >= 0)
      pending_cmd_mode = cm;
    if (cg >= 0)
      pending_cmd_goal = cg;
  }

  if (adcs_sm_every > 0 && completed <= full_steps &&
      completed % static_cast<std::size_t>(adcs_sm_every) == 0) {
    const int prev_am = active_adcs_mode;
    const auto [am, ag] = omni::hardware::step_adcs_sm(
        elapsed_now, adcs_sm_rt, adcs_work, pending_cmd_mode, pending_cmd_goal);
    active_adcs_mode = am;
    active_adcs_goal = ag;
    pending_cmd_mode = -1;
    pending_cmd_goal = -1;
    if (active_adcs_mode != prev_am) {
      apply_adcs_mode_masks(active_adcs_mode, adcs_sm_rt, sensor_enabled,
                            rw_enabled, mtq_enabled);
      hw_dirty = true;
    }
  }

  if (hw_dirty) {
    sampler.set_sensor_enabled(sensor_enabled);
    for (std::size_t i = 0; i < ctrl_sensor_config.size() && i < sensor_enabled.size(); ++i) {
      ctrl_sensor_config[i].enabled = sensor_enabled[i];
    }
    omni::hardware::controllers::sync_actuator_enable(
        rw_enabled, mtq_enabled, base_wheel_max, base_mtq_max, ctrl_rt,
        runtime.wheel_commands, runtime.dipole_commands);
  }

  if (sensor_tick) {
    sampler.sample(y, elapsed_now,
                   completed / static_cast<std::size_t>(sensor_every));
    if (ctrl_rt.enabled) {
      const auto [gq, gw] = omni::hardware::evaluate_goal(
          active_adcs_goal, adcs_sm_rt, y, epoch_now, goal_runtime);
      const std::size_t ctrl_idx =
          adcs_sm_rt.enabled
              ? static_cast<std::size_t>(
                    adcs_sm_rt.mode_default_controller[static_cast<std::size_t>(
                        active_adcs_mode)])
              : default_ctrl_idx;
      if (static_cast<int>(ctrl_idx) != prev_ctrl_idx) {
        ctrl_work.setZero();
        prev_ctrl_idx = static_cast<int>(ctrl_idx);
      }
      const Eigen::Vector4d q_est =
          sampler.has_estimator() ? sampler.estimate().quaternion
                                  : Eigen::Vector4d(y.head<4>());
      const Eigen::Vector3d w_est =
          sampler.has_estimator() ? sampler.estimate().rate
                                  : Eigen::Vector3d(y.segment<3>(4));
      const Eigen::VectorXd h_wheels =
          (y.size() > 13) ? Eigen::VectorXd(y.tail(y.size() - 13))
                          : Eigen::VectorXd::Zero(0);
      std::tie(last_ff_torque, last_ff_q, last_ff_w, last_ref_torque) =
          omni::hardware::controllers::step_controller(
              ctrl_idx, dt * sensor_every, q_est, w_est, gq, gw, h_wheels,
              sampler.readings(), ctrl_sensor_config, ctrl_rt, ctrl_work,
              runtime.wheel_commands, runtime.dipole_commands);
    }
  }
}

void ControlLoopContext::record_row(Eigen::Index row, const Vector &y,
                                    double epoch_jd, const Runtime &runtime) {
  if (record_obc_mode)
    output.at("obc_mode")(row, 0) = static_cast<double>(active_obc_mode);
  if (record_adcs_mode)
    output.at("adcs_mode")(row, 0) = static_cast<double>(active_adcs_mode);
  if (record_adcs_goal)
    output.at("adcs_goal")(row, 0) = static_cast<double>(active_adcs_goal);

  if (ctrl_log_flags & 1) {
    const auto [gq, gw] = omni::hardware::evaluate_goal(
        active_adcs_goal, adcs_sm_rt, y, epoch_jd, goal_runtime);
    if (output.count("goal_attitude"))
      output.at("goal_attitude").row(row) = gq.transpose();
    if (output.count("goal_rate"))
      output.at("goal_rate").row(row) = gw.transpose();
  }
  if (ctrl_log_flags & 2)
    output.at("feedforward_torque").row(row) = last_ff_torque.transpose();
  if (ctrl_log_flags & 4)
    output.at("feedforward_attitude").row(row) = last_ff_q.transpose();
  if (ctrl_log_flags & 8)
    output.at("feedforward_rate").row(row) = last_ff_w.transpose();
  if (ctrl_log_flags & 16)
    output.at("reference_torque").row(row) = last_ref_torque.transpose();
  if ((ctrl_log_flags & 32) && runtime.wheel_commands.size() > 0)
    output.at("commanded_wheel_torque").row(row) =
        runtime.wheel_commands.transpose();
  if ((ctrl_log_flags & 64) && runtime.dipole_commands.size() > 0)
    output.at("commanded_magnetic_dipole").row(row) =
        runtime.dipole_commands.transpose();
}

} // namespace omni::propagation

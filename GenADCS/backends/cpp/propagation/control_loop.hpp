#pragma once

#include "attitude_propagators/simulation_sampling.hpp"
#include "coupled_propagators/_coupled_dynamics.hpp"
#include "../hardware/ADCS_subsystem/controllers/runtime.hpp"
#include "../hardware/ADCS_subsystem/goals/goal.hpp"
#include "../hardware/ADCS_subsystem/state_machine.hpp"
#include "../hardware/OBC_subsystem/runtime.hpp"

#include <Eigen/Core>
#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace omni::propagation {

struct ControlLoopContext {
  bool active = false;

  int obc_every = 0;
  omni::hardware::obc::OBCRuntime obc_rt;
  std::array<double, 5> obc_work{0.0, 0.0, 0.0, 0.0, 0.0};

  int adcs_sm_every = 0;
  omni::hardware::ADCSSMRuntime adcs_sm_rt;
  std::array<int, 3> adcs_work{0, 0, 0};

  omni::hardware::controllers::ControllerRuntime ctrl_rt;
  Eigen::VectorXd ctrl_work;
  std::size_t default_ctrl_idx = 0;
  omni::hardware::controllers::SensorConfig ctrl_sensor_config;
  omni::hardware::goals::GoalRuntime goal_runtime;

  std::vector<bool> sensor_enabled;
  std::vector<bool> rw_enabled;
  std::vector<bool> mtq_enabled;
  Eigen::VectorXd base_wheel_max;
  Eigen::VectorXd base_mtq_max;

  int ctrl_log_flags = 0;
  bool record_obc_mode = false;
  bool record_adcs_mode = false;
  bool record_adcs_goal = false;

  int active_obc_mode = 0;
  int active_adcs_mode = 0;
  int active_adcs_goal = 0;
  int prev_ctrl_idx = -1;
  int pending_cmd_mode = -1;
  int pending_cmd_goal = -1;

  Eigen::Vector3d last_ff_torque;
  Eigen::Vector4d last_ff_q;
  Eigen::Vector3d last_ff_w;
  Eigen::Vector3d last_ref_torque;

  SampleRows output;

  void allocate(Eigen::Index rows, Eigen::Index n_wheels, Eigen::Index n_mtqs,
                const std::set<std::string> &fields);

  void on_init(const Vector &y, double initial_jd, double dt, int sensor_every,
               Runtime &runtime, SimulationSampler &sampler);

  void on_step_pre_record(const Vector &y, std::size_t completed,
                          std::size_t full_steps, double dt, double initial_jd,
                          int sensor_every, bool sensor_tick, Runtime &runtime,
                          SimulationSampler &sampler);

  void record_row(Eigen::Index row, const Vector &y, double epoch_jd,
                  const Runtime &runtime);
};

} // namespace omni::propagation

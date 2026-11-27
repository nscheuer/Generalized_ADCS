#ifndef ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_RUNTIME_HPP
#define ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_RUNTIME_HPP

#include "obc_subsystem.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace omni::hardware::obc {

inline constexpr int FDIR_ANGULAR_RATE = 1;
inline constexpr double OMEGA_EARTH = 7.2921159e-5;

class GoalTable {
public:
  GoalTable();
  int register_goal(const std::shared_ptr<omni::hardware::goals::Goal> &goal);

  std::vector<std::shared_ptr<omni::hardware::goals::Goal>> objects;
  std::vector<std::string> names;
};

struct OBCRuntime {
  bool enabled = false;
  int initial_state = 0;
  int n_states = 0;

  // CSR-indexed telecommand transitions per state:
  std::vector<int> tc_trans_start;
  std::vector<int> tc_trans_end;
  std::vector<int> tc_trans_code;
  std::vector<int> tc_trans_target;
  std::vector<int> tc_trans_goal;

  // CSR-indexed orbital/nadir swath (InFOV) transitions per state:
  std::vector<int> fov_trans_start;
  std::vector<int> fov_trans_end;
  std::vector<Eigen::Vector3d> fov_ecef;
  std::vector<double> fov_cos_limit;
  std::vector<bool> fov_negate;
  std::vector<int> fov_target;
  std::vector<int> fov_goal;

  // Auto-completion / timed transitions per state:
  std::vector<double> state_duration;
  std::vector<int> state_auto_target;

  // ADCS bindings per OBC state (-1 means no override):
  std::vector<int> state_adcs_mode;
  std::vector<int> state_adcs_goal;

  // Hardware enable/disable masks per OBC state (-1 unchanged, 0 off, 1 on):
  std::vector<std::vector<std::int8_t>> state_sensor_mask;
  std::vector<std::vector<std::int8_t>> state_rw_mask;
  std::vector<std::vector<std::int8_t>> state_mtq_mask;
  std::vector<std::vector<std::int8_t>> tc_trans_sensor_mask;
  std::vector<std::vector<std::int8_t>> tc_trans_rw_mask;
  std::vector<std::vector<std::int8_t>> tc_trans_mtq_mask;

  // Sorted telecommand timeline (terminated by inf / -1):
  std::vector<double> tc_times;
  std::vector<int> tc_codes;
  std::vector<int> tc_goals;

  // FDIR configuration:
  bool fdir_enabled = false;
  int fdir_safe_state = 0;
  std::vector<int> fdir_monitor_kinds;
  std::vector<double> fdir_monitor_thresholds;
  std::vector<int> fdir_monitor_targets;

  // CSR-indexed FDIRTriggered transitions per state:
  std::vector<int> fdir_trans_start;
  std::vector<int> fdir_trans_end;
  std::vector<int> fdir_trans_monitor;
  std::vector<int> fdir_trans_target;

  // Human-readable state names:
  std::vector<std::string> mode_names;
};

OBCRuntime prepare_obc(
    const OBCSubsystem *obc,
    const std::map<std::string, int> &adcs_mode_to_id = {},
    GoalTable *goal_table = nullptr,
    const std::vector<std::string> &sensor_names = {},
    const std::vector<std::string> &rw_names = {},
    const std::vector<std::string> &mtq_names = {});

std::array<double, 5> init_obc_work(const OBCRuntime &runtime);

std::tuple<int, int, int, int, int>
step_obc_core(const Eigen::Ref<const Eigen::VectorXd> &y, double elapsed_s,
              double epoch_jd, double dt_obc, const OBCRuntime &obc,
              std::array<double, 5> &work);

std::tuple<int, int, int>
step_obc(const Eigen::Ref<const Eigen::VectorXd> &y, double elapsed_s,
         double epoch_jd, double dt_obc, const OBCRuntime &obc,
         std::array<double, 5> &work);

} // namespace omni::hardware::obc

#endif

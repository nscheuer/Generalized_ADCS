#include "runtime.hpp"

#include "hardware/ADCS_subsystem/goals/coordinate_goal.hpp"
#include "hardware/ADCS_subsystem/goals/no_goal.hpp"
#include "hardware/ADCS_subsystem/goals/quaternion_goal.hpp"
#include "helpers/math/frames/conversions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace omni::hardware::obc {

namespace {

int resolve_target(const std::string &name,
                   const std::map<std::string, const Mode *> &modes_by_name,
                   const std::map<std::string, int> &leaf_ids) {
  auto it = leaf_ids.find(name);
  if (it != leaf_ids.end()) {
    return it->second;
  }
  auto mit = modes_by_name.find(name);
  if (mit != modes_by_name.end() && !mit->second->submodes.empty()) {
    const std::string &sub = *mit->second->initial_submode;
    return leaf_ids.at(name + "." + sub);
  }
  throw std::invalid_argument("unknown target mode: " + name);
}

std::vector<int> resolve_sources(
    const std::string &pattern,
    const std::map<std::string, const Mode *> &modes_by_name,
    const std::vector<std::string> &leaf_names,
    const std::map<std::string, int> &leaf_ids) {
  if (pattern == "*") {
    std::vector<int> all(leaf_names.size());
    for (std::size_t i = 0; i < leaf_names.size(); ++i) {
      all[i] = static_cast<int>(i);
    }
    return all;
  }
  if (pattern.size() >= 2 && pattern.substr(pattern.size() - 2) == ".*") {
    const std::string prefix = pattern.substr(0, pattern.size() - 1);
    std::vector<int> matched;
    for (const auto &n : leaf_names) {
      if (n.rfind(prefix, 0) == 0) {
        matched.push_back(leaf_ids.at(n));
      }
    }
    if (matched.empty()) {
      throw std::invalid_argument("transition source matched no modes: " +
                                  pattern);
    }
    return matched;
  }
  auto it = leaf_ids.find(pattern);
  if (it != leaf_ids.end()) {
    return {it->second};
  }
  auto mit = modes_by_name.find(pattern);
  if (mit != modes_by_name.end() && !mit->second->submodes.empty()) {
    const std::string prefix = pattern + ".";
    std::vector<int> matched;
    for (const auto &n : leaf_names) {
      if (n.rfind(prefix, 0) == 0) {
        matched.push_back(leaf_ids.at(n));
      }
    }
    return matched;
  }
  throw std::invalid_argument("unknown transition source: " + pattern);
}

std::vector<std::int8_t> compile_mask(
    const std::optional<HardwareMaskSpec> &spec,
    const std::vector<std::string> &names,
    const std::vector<std::string> &aliases = {"*", "all", "sensors"},
    const std::vector<std::string> &other_names = {},
    const std::vector<std::string> &other_aliases = {}) {
  std::vector<std::int8_t> mask(names.size(), -1);
  if (!spec.has_value()) {
    return mask;
  }
  auto in_vec = [](const std::vector<std::string> &v, const std::string &k) {
    return std::find(v.begin(), v.end(), k) != v.end();
  };
  for (const auto &[key, val] : *spec) {
    if (in_vec(aliases, key)) {
      std::fill(mask.begin(), mask.end(), val ? 1 : 0);
      continue;
    }
    bool matched = false;
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == key) {
        mask[i] = val ? 1 : 0;
        matched = true;
      }
    }
    if (!matched && !in_vec(other_names, key) && !in_vec(other_aliases, key)) {
      throw std::invalid_argument("unknown hardware reference in state machine: " +
                                  key);
    }
  }
  return mask;
}

OBCRuntime disabled_runtime() {
  OBCRuntime rt;
  rt.enabled = false;
  rt.tc_times = {std::numeric_limits<double>::infinity()};
  rt.tc_codes = {-1};
  rt.tc_goals = {-1};
  return rt;
}

std::tuple<int, int, int> enter_state(int target, double elapsed_s,
                                      const OBCRuntime &obc,
                                      std::array<double, 5> &work,
                                      int goal_override) {
  work[0] = static_cast<double>(target);
  const double dur = obc.state_duration[static_cast<std::size_t>(target)];
  work[2] =
      (dur > 0.0) ? (elapsed_s + dur) : std::numeric_limits<double>::infinity();
  work[4] = elapsed_s;
  const int adcs_m = obc.state_adcs_mode[static_cast<std::size_t>(target)];
  const int adcs_g =
      (goal_override >= 0)
          ? goal_override
          : obc.state_adcs_goal[static_cast<std::size_t>(target)];
  return {target, adcs_m, adcs_g};
}

std::string goal_type_name(const omni::hardware::goals::Goal &goal) {
  if (dynamic_cast<const omni::hardware::goals::NoGoal *>(&goal) != nullptr) {
    return "NoGoal";
  }
  if (dynamic_cast<const omni::hardware::goals::QuaternionGoal *>(&goal) !=
      nullptr) {
    return "QuaternionGoal";
  }
  if (dynamic_cast<const omni::hardware::goals::CoordinateGoal *>(&goal) !=
      nullptr) {
    return "CoordinateGoal";
  }
  return "Goal";
}

} // namespace

GoalTable::GoalTable() {
  objects.push_back(std::make_shared<omni::hardware::goals::NoGoal>());
  names.push_back("NoGoal");
}

int GoalTable::register_goal(
    const std::shared_ptr<omni::hardware::goals::Goal> &goal) {
  if (!goal ||
      dynamic_cast<const omni::hardware::goals::NoGoal *>(goal.get()) !=
          nullptr) {
    return 0;
  }
  const int idx = static_cast<int>(objects.size());
  objects.push_back(goal);
  names.push_back(goal_type_name(*goal));
  return idx;
}

OBCRuntime prepare_obc(const OBCSubsystem *obc,
                       const std::map<std::string, int> &adcs_mode_to_id,
                       GoalTable *goal_table,
                       const std::vector<std::string> &sensor_names,
                       const std::vector<std::string> &rw_names,
                       const std::vector<std::string> &mtq_names) {
  if (obc == nullptr || !obc->state_machine) {
    return disabled_runtime();
  }

  GoalTable local_table;
  GoalTable &gt = (goal_table != nullptr) ? *goal_table : local_table;

  const StateMachine &sm = *obc->state_machine;
  std::map<std::string, const Mode *> modes_by_name;
  for (const auto &m : sm.modes) {
    modes_by_name[m.name] = &m;
  }

  std::vector<std::string> leaf_names;
  std::vector<const Mode *> leaf_modes;
  std::vector<const Mode *> leaf_parents;
  for (const auto &m : sm.modes) {
    if (!m.submodes.empty()) {
      for (const auto &sub : m.submodes) {
        leaf_names.push_back(m.name + "." + sub.name);
        leaf_modes.push_back(&sub);
        leaf_parents.push_back(&m);
      }
    } else {
      leaf_names.push_back(m.name);
      leaf_modes.push_back(&m);
      leaf_parents.push_back(nullptr);
    }
  }

  std::map<std::string, int> leaf_ids;
  for (std::size_t i = 0; i < leaf_names.size(); ++i) {
    leaf_ids[leaf_names[i]] = static_cast<int>(i);
  }
  const int n_states = static_cast<int>(leaf_names.size());
  const int initial_state = resolve_target(sm.initial, modes_by_name, leaf_ids);

  std::map<std::string, int> cmd_to_code;
  auto get_cmd_code = [&](const std::string &cmd) -> int {
    auto it = cmd_to_code.find(cmd);
    if (it != cmd_to_code.end()) {
      return it->second;
    }
    const int code = static_cast<int>(cmd_to_code.size());
    cmd_to_code[cmd] = code;
    return code;
  };

  OBCRuntime rt;
  rt.enabled = true;
  rt.initial_state = initial_state;
  rt.n_states = n_states;
  rt.mode_names = leaf_names;

  const auto ns = static_cast<std::size_t>(n_states);
  rt.state_duration.assign(ns, -1.0);
  rt.state_auto_target.assign(ns, -1);
  rt.state_adcs_mode.assign(ns, -1);
  rt.state_adcs_goal.assign(ns, -1);
  rt.state_sensor_mask.assign(ns, std::vector<std::int8_t>(sensor_names.size(), -1));
  rt.state_rw_mask.assign(ns, std::vector<std::int8_t>(rw_names.size(), -1));
  rt.state_mtq_mask.assign(ns, std::vector<std::int8_t>(mtq_names.size(), -1));

  for (int idx = 0; idx < n_states; ++idx) {
    const auto i = static_cast<std::size_t>(idx);
    const Mode *leaf = leaf_modes[i];
    const Mode *parent = leaf_parents[i];

    if (leaf->duration.has_value()) {
      rt.state_duration[i] = *leaf->duration;
      if (leaf->auto_target.has_value()) {
        std::string target_name = *leaf->auto_target;
        if (parent != nullptr &&
            target_name.find('.') == std::string::npos &&
            leaf_ids.count(parent->name + "." + target_name) > 0) {
          target_name = parent->name + "." + target_name;
        }
        rt.state_auto_target[i] =
            resolve_target(target_name, modes_by_name, leaf_ids);
      }
    }

    std::optional<ADCSBinding> binding = leaf->bindings.adcs;
    if (!binding.has_value() && parent != nullptr) {
      binding = parent->bindings.adcs;
    }
    if (binding.has_value()) {
      if (const auto *mode_name = std::get_if<std::string>(&*binding)) {
        auto it = adcs_mode_to_id.find(*mode_name);
        if (it == adcs_mode_to_id.end()) {
          throw std::invalid_argument(
              "OBC binding references unknown ADCS mode: " + *mode_name);
        }
        rt.state_adcs_mode[i] = it->second;
      } else if (const auto *goal_ptr =
                     std::get_if<std::shared_ptr<omni::hardware::goals::Goal>>(
                         &*binding)) {
        rt.state_adcs_goal[i] = gt.register_goal(*goal_ptr);
      } else if (const auto *pair_val = std::get_if<
                     std::pair<std::string,
                               std::shared_ptr<omni::hardware::goals::Goal>>>(
                     &*binding)) {
        auto it = adcs_mode_to_id.find(pair_val->first);
        if (it == adcs_mode_to_id.end()) {
          throw std::invalid_argument(
              "OBC binding references unknown ADCS mode: " + pair_val->first);
        }
        rt.state_adcs_mode[i] = it->second;
        rt.state_adcs_goal[i] = gt.register_goal(pair_val->second);
      }
    }

    auto sensor_binding = leaf->bindings.sensors;
    if (!sensor_binding.has_value() && parent != nullptr) {
      sensor_binding = parent->bindings.sensors;
    }
    if (sensor_binding.has_value()) {
      rt.state_sensor_mask[i] = compile_mask(sensor_binding, sensor_names);
    }

    auto actuator_binding = leaf->bindings.actuators;
    if (!actuator_binding.has_value() && parent != nullptr) {
      actuator_binding = parent->bindings.actuators;
    }
    if (actuator_binding.has_value()) {
      rt.state_rw_mask[i] = compile_mask(
          actuator_binding, rw_names,
          {"*", "all", "actuators", "reaction_wheels", "ReactionWheel"},
          mtq_names, {"magnetorquers", "Magnetorquer"});
      rt.state_mtq_mask[i] = compile_mask(
          actuator_binding, mtq_names,
          {"*", "all", "actuators", "magnetorquers", "Magnetorquer"},
          rw_names, {"reaction_wheels", "ReactionWheel"});
    }
  }

  std::map<std::string, int> monitor_to_id;
  if (obc->fdir) {
    for (std::size_t i = 0; i < obc->fdir->monitors().size(); ++i) {
      monitor_to_id[obc->fdir->monitors()[i]->name()] = static_cast<int>(i);
    }
  }

  struct TCEntry {
    int priority;
    int code;
    int target;
    int goal;
    std::vector<std::int8_t> s_mask;
    std::vector<std::int8_t> rw_mask;
    std::vector<std::int8_t> mtq_mask;
  };
  struct FOVEntry {
    int priority;
    Eigen::Vector3d ecef;
    double cos_limit;
    bool negate;
    int target;
    int goal;
  };
  struct FDIREntry {
    int priority;
    int mon_idx;
    int target;
  };

  std::vector<std::vector<TCEntry>> per_state_tc(ns);
  std::vector<std::vector<FOVEntry>> per_state_fov(ns);
  std::vector<std::vector<FDIREntry>> per_state_fdir(ns);

  for (const auto &tr : sm.transitions) {
    const auto sources =
        resolve_sources(tr.source, modes_by_name, leaf_names, leaf_ids);
    const int target_id = resolve_target(tr.target, modes_by_name, leaf_ids);

    if (const auto *tc = dynamic_cast<const Telecommand *>(tr.trigger.get())) {
      const int code = get_cmd_code(tc->command());
      const int g_idx = tc->goal() ? gt.register_goal(tc->goal()) : -1;
      const auto s_m = compile_mask(tc->sensors(), sensor_names);
      const auto rw_m = compile_mask(
          tc->actuators(), rw_names,
          {"*", "all", "actuators", "reaction_wheels", "ReactionWheel"},
          mtq_names, {"magnetorquers", "Magnetorquer"});
      const auto mtq_m = compile_mask(
          tc->actuators(), mtq_names,
          {"*", "all", "actuators", "magnetorquers", "Magnetorquer"},
          rw_names, {"reaction_wheels", "ReactionWheel"});
      for (int src : sources) {
        per_state_tc[static_cast<std::size_t>(src)].push_back(
            TCEntry{tr.priority, code, target_id, g_idx, s_m, rw_m, mtq_m});
      }
    } else if (const auto *fov = dynamic_cast<const InFOV *>(tr.trigger.get())) {
      auto coord = std::make_shared<omni::hardware::goals::CoordinateGoal>(
          fov->lat(), fov->lon(), fov->alt());
      const double pi = std::acos(-1.0);
      const double cos_limit = std::cos(fov->max_off_nadir_deg() * pi / 180.0);
      int g_idx = -1;
      if (fov->goal()) {
        g_idx = gt.register_goal(fov->goal());
      } else if (!fov->negate()) {
        g_idx = gt.register_goal(coord);
      }
      for (int src : sources) {
        per_state_fov[static_cast<std::size_t>(src)].push_back(
            FOVEntry{tr.priority, coord->target_ecef_m(), cos_limit,
                     fov->negate(), target_id, g_idx});
      }
    } else if (dynamic_cast<const Completion *>(tr.trigger.get()) != nullptr) {
      for (int src : sources) {
        const auto s = static_cast<std::size_t>(src);
        if (rt.state_duration[s] <= 0.0) {
          throw std::invalid_argument(
              "Completion transition requires Mode.duration for " +
              leaf_names[s]);
        }
        rt.state_auto_target[s] = target_id;
      }
    } else if (const auto *after =
                   dynamic_cast<const After *>(tr.trigger.get())) {
      for (int src : sources) {
        const auto s = static_cast<std::size_t>(src);
        rt.state_duration[s] = after->duration();
        rt.state_auto_target[s] = target_id;
      }
    } else if (const auto *fdir_tr =
                   dynamic_cast<const FDIRTriggered *>(tr.trigger.get())) {
      int mon_idx = -1;
      if (fdir_tr->monitor().has_value()) {
        auto mit = monitor_to_id.find(*fdir_tr->monitor());
        if (mit == monitor_to_id.end()) {
          throw std::invalid_argument(
              "FDIRTriggered references unknown monitor: " +
              *fdir_tr->monitor());
        }
        mon_idx = mit->second;
      }
      for (int src : sources) {
        per_state_fdir[static_cast<std::size_t>(src)].push_back(
            FDIREntry{tr.priority, mon_idx, target_id});
      }
    }
  }

  rt.tc_trans_start.resize(ns);
  rt.tc_trans_end.resize(ns);
  int cursor = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    rt.tc_trans_start[s] = cursor;
    std::stable_sort(per_state_tc[s].begin(), per_state_tc[s].end(),
                     [](const TCEntry &a, const TCEntry &b) {
                       return a.priority > b.priority;
                     });
    for (const auto &e : per_state_tc[s]) {
      rt.tc_trans_code.push_back(e.code);
      rt.tc_trans_target.push_back(e.target);
      rt.tc_trans_goal.push_back(e.goal);
      rt.tc_trans_sensor_mask.push_back(e.s_mask);
      rt.tc_trans_rw_mask.push_back(e.rw_mask);
      rt.tc_trans_mtq_mask.push_back(e.mtq_mask);
      ++cursor;
    }
    rt.tc_trans_end[s] = cursor;
  }

  rt.fov_trans_start.resize(ns);
  rt.fov_trans_end.resize(ns);
  cursor = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    rt.fov_trans_start[s] = cursor;
    std::stable_sort(per_state_fov[s].begin(), per_state_fov[s].end(),
                     [](const FOVEntry &a, const FOVEntry &b) {
                       return a.priority > b.priority;
                     });
    for (const auto &e : per_state_fov[s]) {
      rt.fov_ecef.push_back(e.ecef);
      rt.fov_cos_limit.push_back(e.cos_limit);
      rt.fov_negate.push_back(e.negate);
      rt.fov_target.push_back(e.target);
      rt.fov_goal.push_back(e.goal);
      ++cursor;
    }
    rt.fov_trans_end[s] = cursor;
  }

  rt.fdir_trans_start.resize(ns);
  rt.fdir_trans_end.resize(ns);
  cursor = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    rt.fdir_trans_start[s] = cursor;
    std::stable_sort(per_state_fdir[s].begin(), per_state_fdir[s].end(),
                     [](const FDIREntry &a, const FDIREntry &b) {
                       return a.priority > b.priority;
                     });
    for (const auto &e : per_state_fdir[s]) {
      rt.fdir_trans_monitor.push_back(e.mon_idx);
      rt.fdir_trans_target.push_back(e.target);
      ++cursor;
    }
    rt.fdir_trans_end[s] = cursor;
  }

  std::vector<TimelineEntry> combined = sm.timeline;
  combined.insert(combined.end(), obc->timeline.begin(), obc->timeline.end());
  std::stable_sort(combined.begin(), combined.end(),
                   [](const TimelineEntry &a, const TimelineEntry &b) {
                     return std::get<0>(a) < std::get<0>(b);
                   });
  for (const auto &entry : combined) {
    rt.tc_times.push_back(std::get<0>(entry));
    rt.tc_codes.push_back(get_cmd_code(std::get<1>(entry)));
    const auto &g = std::get<2>(entry);
    rt.tc_goals.push_back(g ? gt.register_goal(g) : -1);
  }
  rt.tc_times.push_back(std::numeric_limits<double>::infinity());
  rt.tc_codes.push_back(-1);
  rt.tc_goals.push_back(-1);

  if (obc->fdir && !obc->fdir->monitors().empty()) {
    rt.fdir_enabled = true;
    rt.fdir_safe_state =
        resolve_target(obc->fdir->safe_mode(), modes_by_name, leaf_ids);
    for (const auto &mon : obc->fdir->monitors()) {
      const auto *arm = dynamic_cast<const AngularRateMonitor *>(mon.get());
      if (arm == nullptr) {
        throw std::invalid_argument("unsupported FDIR monitor");
      }
      rt.fdir_monitor_kinds.push_back(FDIR_ANGULAR_RATE);
      rt.fdir_monitor_thresholds.push_back(arm->max_rate());
      rt.fdir_monitor_targets.push_back(
          arm->target_mode().has_value()
              ? resolve_target(*arm->target_mode(), modes_by_name, leaf_ids)
              : rt.fdir_safe_state);
    }
  }

  return rt;
}

std::array<double, 5> init_obc_work(const OBCRuntime &runtime) {
  double auto_t = std::numeric_limits<double>::infinity();
  if (runtime.enabled &&
      runtime.state_duration[static_cast<std::size_t>(runtime.initial_state)] >
          0.0) {
    auto_t =
        runtime.state_duration[static_cast<std::size_t>(runtime.initial_state)];
  }
  return {static_cast<double>(runtime.initial_state), 0.0, auto_t, 0.0, 0.0};
}

std::tuple<int, int, int, int, int>
step_obc_core(const Eigen::Ref<const Eigen::VectorXd> &y, double elapsed_s,
              double epoch_jd, double dt_obc, const OBCRuntime &obc,
              std::array<double, 5> &work) {
  int state = static_cast<int>(work[0]);
  int cmd_mode = -1;
  int cmd_goal = -1;
  int entered_state = -1;
  int fired_tc_idx = -1;

  // 1. Fast FDIR check.
  if (obc.fdir_enabled && work[3] == 0.0) {
    for (std::size_t i = 0; i < obc.fdir_monitor_kinds.size(); ++i) {
      if (obc.fdir_monitor_kinds[i] == FDIR_ANGULAR_RATE) {
        const double wx = y[4];
        const double wy = y[5];
        const double wz = y[6];
        const double limit = obc.fdir_monitor_thresholds[i];
        if (wx * wx + wy * wy + wz * wz > limit * limit) {
          work[3] = 1.0;
          int fdir_target = obc.fdir_monitor_targets[i];
          const int f_start =
              obc.fdir_trans_start[static_cast<std::size_t>(state)];
          const int f_end =
              obc.fdir_trans_end[static_cast<std::size_t>(state)];
          for (int k = f_start; k < f_end; ++k) {
            const int mon_filter =
                obc.fdir_trans_monitor[static_cast<std::size_t>(k)];
            if (mon_filter < 0 || mon_filter == static_cast<int>(i)) {
              fdir_target = obc.fdir_trans_target[static_cast<std::size_t>(k)];
              break;
            }
          }
          const auto [st, cm, cg] =
              enter_state(fdir_target, elapsed_s, obc, work, -1);
          return {st, cm, cg, st, -1};
        }
      }
    }
  }

  // 2. Telecommand & auto-completion wake-up check.
  auto tc_idx = static_cast<std::size_t>(work[1]);
  const double next_tc_t = obc.tc_times[tc_idx];
  const double auto_t = work[2];
  if (elapsed_s >= next_tc_t || elapsed_s >= auto_t) {
    while (elapsed_s >= obc.tc_times[tc_idx]) {
      const int cmd = obc.tc_codes[tc_idx];
      const int tl_goal = obc.tc_goals[tc_idx];
      ++tc_idx;
      work[1] = static_cast<double>(tc_idx);
      const int start = obc.tc_trans_start[static_cast<std::size_t>(state)];
      const int end = obc.tc_trans_end[static_cast<std::size_t>(state)];
      for (int k = start; k < end; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        if (obc.tc_trans_code[ku] == cmd) {
          const int target = obc.tc_trans_target[ku];
          if (target != state) {
            work[3] = 0.0;
          }
          const int tr_goal = obc.tc_trans_goal[ku];
          const int override_goal = (tl_goal >= 0) ? tl_goal : tr_goal;
          const auto [st, cm, cg] =
              enter_state(target, elapsed_s, obc, work, override_goal);
          state = st;
          entered_state = state;
          fired_tc_idx = k;
          if (cm >= 0) {
            cmd_mode = cm;
          }
          if (cg >= 0) {
            cmd_goal = cg;
          }
          break;
        }
      }
    }

    if (elapsed_s >= work[2]) {
      const int auto_target =
          obc.state_auto_target[static_cast<std::size_t>(state)];
      if (auto_target >= 0) {
        if (auto_target != state) {
          work[3] = 0.0;
        }
        const auto [st, cm, cg] =
            enter_state(auto_target, elapsed_s, obc, work, -1);
        state = st;
        entered_state = state;
        if (cm >= 0) {
          cmd_mode = cm;
        }
        if (cg >= 0) {
          cmd_goal = cg;
        }
      }
    }
  }

  // 3. Active-state orbital/nadir swath (InFOV) check with spatial skip timer.
  const int fov_start = obc.fov_trans_start[static_cast<std::size_t>(state)];
  const int fov_end = obc.fov_trans_end[static_cast<std::size_t>(state)];
  if (fov_start < fov_end && elapsed_s >= work[4]) {
    const Eigen::Vector3d r_sat = y.segment<3>(7);
    const Eigen::Vector3d v_sat = y.segment<3>(10);
    const double r_norm = r_sat.norm();
    const double v_norm = v_sat.norm();
    const Eigen::Matrix3d rot =
        omni::frames::ecef_to_gcrf_matrix(epoch_jd, epoch_jd);
    double min_skip = std::numeric_limits<double>::infinity();

    for (int k = fov_start; k < fov_end; ++k) {
      const auto ku = static_cast<std::size_t>(k);
      const Eigen::Vector3d &t_ecef = obc.fov_ecef[ku];
      const Eigen::Vector3d t_eci = rot * t_ecef;
      const Eigen::Vector3d rel = t_eci - r_sat;
      const double dist = rel.norm();
      const double rt_sq = t_ecef.squaredNorm();
      const double r_horizon = std::sqrt(std::max(0.0, r_norm * r_norm - rt_sq));

      bool in_cone = false;
      if (dist > r_horizon) {
        in_cone = false;
        const double v_max = v_norm + OMEGA_EARTH * std::sqrt(rt_sq);
        const double dt_skip = (dist - r_horizon) / std::max(1.0, v_max);
        if (0.9 * dt_skip < min_skip) {
          min_skip = 0.9 * dt_skip;
        }
      } else {
        const double cos_off_nadir = -rel.dot(r_sat) / (dist * r_norm);
        in_cone = cos_off_nadir >= obc.fov_cos_limit[ku];
        min_skip = 0.0;
      }

      const bool triggered = obc.fov_negate[ku] ? (!in_cone) : in_cone;
      if (triggered) {
        const int fov_target = obc.fov_target[ku];
        if (fov_target != state) {
          work[3] = 0.0;
        }
        const auto [st, cm, cg] = enter_state(fov_target, elapsed_s, obc, work,
                                              obc.fov_goal[ku]);
        state = st;
        entered_state = state;
        if (cm >= 0) {
          cmd_mode = cm;
        }
        if (cg >= 0) {
          cmd_goal = cg;
        }
        return {state, cmd_mode, cmd_goal, entered_state, fired_tc_idx};
      }
    }

    if (min_skip > dt_obc && std::isfinite(min_skip)) {
      work[4] = elapsed_s + min_skip;
    } else {
      work[4] = elapsed_s + dt_obc;
    }
  }

  return {state, cmd_mode, cmd_goal, entered_state, fired_tc_idx};
}

std::tuple<int, int, int> step_obc(const Eigen::Ref<const Eigen::VectorXd> &y,
                                   double elapsed_s, double epoch_jd,
                                   double dt_obc, const OBCRuntime &obc,
                                   std::array<double, 5> &work) {
  const auto [state, cmd_mode, cmd_goal, entered, tc_idx] =
      step_obc_core(y, elapsed_s, epoch_jd, dt_obc, obc, work);
  (void)entered;
  (void)tc_idx;
  return {state, cmd_mode, cmd_goal};
}

} // namespace omni::hardware::obc

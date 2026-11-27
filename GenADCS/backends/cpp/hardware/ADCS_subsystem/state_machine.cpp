#include "state_machine.hpp"

#include "goals/along_track_goal.hpp"
#include "goals/angular_rate_goal.hpp"
#include "goals/coordinate_fmc_goal.hpp"
#include "goals/coordinate_goal.hpp"
#include "goals/cross_track_goal.hpp"
#include "goals/nadir_goal.hpp"
#include "goals/quaternion_goal.hpp"
#include "goals/sun_goal.hpp"
#include "goals/velocity_goal.hpp"
#include "goals/zenith_goal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omni::hardware {

GoalTable::GoalTable() {
  kinds.push_back(GOAL_NONE);
  params.push_back(Eigen::Vector4d::Zero());
  names.push_back("NoGoal");
  objects.push_back(std::make_shared<goals::NoGoal>());
}

int GoalTable::register_goal(const std::shared_ptr<goals::Goal> &goal) {
  if (!goal || dynamic_cast<const goals::NoGoal *>(goal.get()) != nullptr) {
    return 0;
  }
  Eigen::Vector4d p = Eigen::Vector4d::Zero();
  int kind = GOAL_NONE;
  std::string label;

  if (dynamic_cast<const goals::NadirGoal *>(goal.get()) != nullptr) {
    kind = GOAL_NADIR;
    label = "NadirGoal";
  } else if (dynamic_cast<const goals::ZenithGoal *>(goal.get()) != nullptr) {
    kind = GOAL_ZENITH;
    label = "ZenithGoal";
  } else if (dynamic_cast<const goals::AlongTrackGoal *>(goal.get()) !=
             nullptr) {
    kind = GOAL_ALONG_TRACK;
    label = "AlongTrackGoal";
  } else if (dynamic_cast<const goals::CrossTrackGoal *>(goal.get()) !=
             nullptr) {
    kind = GOAL_CROSS_TRACK;
    label = "CrossTrackGoal";
  } else if (dynamic_cast<const goals::VelocityGoal *>(goal.get()) != nullptr) {
    kind = GOAL_VELOCITY;
    label = "VelocityGoal";
  } else if (dynamic_cast<const goals::SunGoal *>(goal.get()) != nullptr) {
    kind = GOAL_SUN;
    label = "SunGoal";
  } else if (const auto *cfmc =
                 dynamic_cast<const goals::CoordinateFMCGoal *>(goal.get())) {
    kind = GOAL_COORDINATE_FMC;
    p.head<3>() = cfmc->target_ecef_m();
    p[3] = static_cast<double>(cfmc->fmc_level());
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "CoordinateFMCGoal("
        << cfmc->lat_deg() << ", " << cfmc->lon_deg()
        << ", fmc=" << cfmc->fmc_level() << ")";
    label = oss.str();
  } else if (const auto *coord =
                 dynamic_cast<const goals::CoordinateGoal *>(goal.get())) {
    kind = GOAL_COORDINATE;
    p.head<3>() = coord->target_ecef_m();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "CoordinateGoal("
        << coord->lat_deg() << ", " << coord->lon_deg() << ")";
    label = oss.str();
  } else if (const auto *quat =
                 dynamic_cast<const goals::QuaternionGoal *>(goal.get())) {
    kind = GOAL_QUATERNION;
    p = quat->quaternion();
    label = "QuaternionGoal";
  } else if (const auto *rate =
                 dynamic_cast<const goals::AngularRateGoal *>(goal.get())) {
    kind = GOAL_ANGULAR_RATE;
    p.head<3>() = rate->rate();
    label = "AngularRateGoal";
  } else {
    throw std::invalid_argument("unsupported Goal in ADCS state machine");
  }

  const int idx = static_cast<int>(kinds.size());
  kinds.push_back(kind);
  params.push_back(p);
  names.push_back(std::move(label));
  objects.push_back(goal);
  return idx;
}

std::vector<std::int8_t>
compile_sensor_mask(const DeviceMaskSpec &spec,
                    const std::vector<std::string> &sensor_names) {
  std::vector<std::int8_t> mask(sensor_names.size(), -1);
  if (spec.empty() || sensor_names.empty()) {
    return mask;
  }
  for (const auto &[key, val] : spec) {
    bool matched = false;
    const auto code = static_cast<std::int8_t>(val ? 1 : 0);
    for (std::size_t i = 0; i < sensor_names.size(); ++i) {
      if (key == "*" || key == "all" || key == "sensors" ||
          key == sensor_names[i]) {
        mask[i] = code;
        matched = true;
      }
    }
    if (!matched) {
      throw std::invalid_argument(
          "unknown sensor reference in state machine: " + key);
    }
  }
  return mask;
}

std::pair<std::vector<std::int8_t>, std::vector<std::int8_t>>
compile_actuator_masks(const DeviceMaskSpec &spec,
                       const std::vector<std::string> &rw_names,
                       const std::vector<std::string> &mtq_names) {
  std::vector<std::int8_t> rw_mask(rw_names.size(), -1);
  std::vector<std::int8_t> mtq_mask(mtq_names.size(), -1);
  if (spec.empty() || (rw_names.empty() && mtq_names.empty())) {
    return {rw_mask, mtq_mask};
  }
  for (const auto &[key, val] : spec) {
    bool matched = false;
    const auto code = static_cast<std::int8_t>(val ? 1 : 0);
    for (std::size_t i = 0; i < rw_names.size(); ++i) {
      if (key == "*" || key == "all" || key == "actuators" ||
          key == "reaction_wheels" || key == "ReactionWheel" ||
          key == rw_names[i]) {
        rw_mask[i] = code;
        matched = true;
      }
    }
    for (std::size_t i = 0; i < mtq_names.size(); ++i) {
      if (key == "*" || key == "all" || key == "actuators" ||
          key == "magnetorquers" || key == "Magnetorquer" ||
          key == mtq_names[i]) {
        mtq_mask[i] = code;
        matched = true;
      }
    }
    if (!matched) {
      throw std::invalid_argument(
          "unknown actuator reference in state machine: " + key);
    }
  }
  return {rw_mask, mtq_mask};
}

void apply_hw_mask(const std::vector<std::int8_t> &mask,
                   std::vector<bool> &enabled) {
  for (std::size_t i = 0; i < enabled.size() && i < mask.size(); ++i) {
    if (mask[i] == 0) {
      enabled[i] = false;
    } else if (mask[i] == 1) {
      enabled[i] = true;
    }
  }
}

ADCSMode::ADCSMode(std::string name_val, std::shared_ptr<goals::Goal> goal_val,
                   std::shared_ptr<controllers::Controller> controller_val,
                   DeviceMaskSpec sensors_val, DeviceMaskSpec actuators_val)
    : name(std::move(name_val)),
      goal(goal_val ? std::move(goal_val) : std::make_shared<goals::NoGoal>()),
      controller(controller_val
                     ? std::move(controller_val)
                     : std::make_shared<controllers::NoController>()),
      sensors(std::move(sensors_val)), actuators(std::move(actuators_val)) {
  if (name.empty()) {
    throw std::invalid_argument("ADCSMode name must be nonempty");
  }
}

ADCSStateMachine::ADCSStateMachine(std::string initial_val,
                                   std::vector<ADCSMode> modes_val,
                                   std::vector<ADCSTimelineEntry> timeline_val)
    : initial(std::move(initial_val)), modes(std::move(modes_val)),
      timeline(std::move(timeline_val)) {
  if (modes.empty()) {
    throw std::invalid_argument(
        "ADCSStateMachine requires at least one ADCSMode");
  }
  std::set<std::string> names;
  for (const auto &m : modes) {
    if (!names.insert(m.name).second) {
      throw std::invalid_argument("duplicate ADCSMode names");
    }
  }
  if (names.find(initial) == names.end()) {
    throw std::invalid_argument("initial ADCS mode not found: " + initial);
  }
  for (const auto &[t, item] : timeline) {
    if (!std::isfinite(t) || t < 0.0) {
      throw std::invalid_argument(
          "ADCS timeline times must be finite and nonnegative");
    }
    if (const auto *mode_name = std::get_if<std::string>(&item)) {
      if (names.find(*mode_name) == names.end()) {
        throw std::invalid_argument("unknown ADCS mode in timeline: " +
                                    *mode_name);
      }
    }
  }
}

ADCSSMRuntime prepare_adcs_sm(const ADCSStateMachine *adcs_sm,
                              GoalTable *goal_table,
                              controllers::ControllerTable *controller_table,
                              const std::vector<std::string> &sensor_names,
                              const std::vector<std::string> &rw_names,
                              const std::vector<std::string> &mtq_names) {
  GoalTable local_gt;
  GoalTable &gt = (goal_table != nullptr) ? *goal_table : local_gt;
  controllers::ControllerTable local_ct;
  controllers::ControllerTable &ct =
      (controller_table != nullptr) ? *controller_table : local_ct;

  ADCSSMRuntime rt;
  if (adcs_sm == nullptr) {
    rt.enabled = false;
    rt.initial_mode = 0;
    rt.mode_default_goal = {0};
    rt.mode_default_controller = {0};
    rt.mode_sensor_mask = {std::vector<std::int8_t>(sensor_names.size(), -1)};
    rt.mode_rw_mask = {std::vector<std::int8_t>(rw_names.size(), -1)};
    rt.mode_mtq_mask = {std::vector<std::int8_t>(mtq_names.size(), -1)};
    rt.tl_times = {std::numeric_limits<double>::infinity()};
    rt.tl_modes = {-1};
    rt.tl_goals = {-1};
    rt.goal_kinds = gt.kinds;
    rt.goal_params = gt.params;
    rt.goal_objects = gt.objects;
    rt.mode_names = {"default"};
    rt.goal_names = gt.names;
    return rt;
  }

  std::map<std::string, int> mode_to_id;
  const std::size_t n_modes = adcs_sm->modes.size();
  rt.mode_names.reserve(n_modes);
  rt.mode_default_goal.resize(n_modes);
  rt.mode_default_controller.resize(n_modes);
  rt.mode_sensor_mask.assign(n_modes,
                             std::vector<std::int8_t>(sensor_names.size(), -1));
  rt.mode_rw_mask.assign(n_modes, std::vector<std::int8_t>(rw_names.size(), -1));
  rt.mode_mtq_mask.assign(n_modes,
                          std::vector<std::int8_t>(mtq_names.size(), -1));

  for (std::size_t idx = 0; idx < n_modes; ++idx) {
    const auto &m = adcs_sm->modes[idx];
    mode_to_id[m.name] = static_cast<int>(idx);
    rt.mode_names.push_back(m.name);
    rt.mode_default_goal[idx] = gt.register_goal(m.goal);
    rt.mode_default_controller[idx] =
        static_cast<int>(ct.register_controller(m.controller));
    if (!m.sensors.empty()) {
      rt.mode_sensor_mask[idx] = compile_sensor_mask(m.sensors, sensor_names);
    }
    if (!m.actuators.empty()) {
      std::tie(rt.mode_rw_mask[idx], rt.mode_mtq_mask[idx]) =
          compile_actuator_masks(m.actuators, rw_names, mtq_names);
    }
  }

  std::vector<ADCSTimelineEntry> sorted_tl = adcs_sm->timeline;
  std::stable_sort(sorted_tl.begin(), sorted_tl.end(),
                   [](const ADCSTimelineEntry &a, const ADCSTimelineEntry &b) {
                     return a.first < b.first;
                   });
  for (const auto &[t, item] : sorted_tl) {
    rt.tl_times.push_back(t);
    if (const auto *mode_name = std::get_if<std::string>(&item)) {
      const int mid = mode_to_id.at(*mode_name);
      rt.tl_modes.push_back(mid);
      rt.tl_goals.push_back(rt.mode_default_goal[static_cast<std::size_t>(mid)]);
    } else {
      const auto &g_ptr = std::get<std::shared_ptr<goals::Goal>>(item);
      rt.tl_modes.push_back(-1);
      rt.tl_goals.push_back(gt.register_goal(g_ptr));
    }
  }
  rt.tl_times.push_back(std::numeric_limits<double>::infinity());
  rt.tl_modes.push_back(-1);
  rt.tl_goals.push_back(-1);

  rt.enabled = true;
  rt.initial_mode = mode_to_id.at(adcs_sm->initial);
  rt.goal_kinds = gt.kinds;
  rt.goal_params = gt.params;
  rt.goal_objects = gt.objects;
  rt.goal_names = gt.names;
  return rt;
}

std::array<int, 3> init_adcs_work(const ADCSSMRuntime &runtime) {
  return {runtime.initial_mode,
          runtime.mode_default_goal[static_cast<std::size_t>(
              runtime.initial_mode)],
          0};
}

std::pair<int, int> step_adcs_sm(double elapsed_s, const ADCSSMRuntime &adcs_sm,
                                 std::array<int, 3> &work, int cmd_mode,
                                 int cmd_goal) {
  int mode = work[0];
  int goal = work[1];

  if (cmd_mode >= 0 && cmd_mode != mode) {
    mode = cmd_mode;
    goal = adcs_sm.mode_default_goal[static_cast<std::size_t>(mode)];
  }
  if (cmd_goal >= 0) {
    goal = cmd_goal;
  }

  auto tl_idx = static_cast<std::size_t>(work[2]);
  while (elapsed_s >= adcs_sm.tl_times[tl_idx]) {
    const int next_m = adcs_sm.tl_modes[tl_idx];
    const int next_g = adcs_sm.tl_goals[tl_idx];
    if (next_m >= 0) {
      mode = next_m;
    }
    if (next_g >= 0) {
      goal = next_g;
    }
    ++tl_idx;
  }
  work[2] = static_cast<int>(tl_idx);
  work[0] = mode;
  work[1] = goal;
  return {mode, goal};
}

std::pair<Eigen::Vector4d, Eigen::Vector3d>
evaluate_goal(int goal_idx, const ADCSSMRuntime &adcs_sm,
              const Eigen::Ref<const Eigen::VectorXd> &state, double epoch_jd,
              const goals::GoalRuntime &runtime) {
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Vector4d nan_q = Eigen::Vector4d::Constant(nan_v);
  const Eigen::Vector3d nan_w = Eigen::Vector3d::Constant(nan_v);
  if (goal_idx <= 0 ||
      static_cast<std::size_t>(goal_idx) >= adcs_sm.goal_objects.size()) {
    return {nan_q, nan_w};
  }
  const auto &[q_opt, w_opt] =
      adcs_sm.goal_objects[static_cast<std::size_t>(goal_idx)]->reference(
          state, epoch_jd, runtime);
  return {q_opt.value_or(nan_q), w_opt.value_or(nan_w)};
}

} // namespace omni::hardware

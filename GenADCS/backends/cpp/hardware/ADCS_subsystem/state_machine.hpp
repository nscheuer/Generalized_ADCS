#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_STATE_MACHINE_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_STATE_MACHINE_HPP

#include "controllers/controller.hpp"
#include "controllers/runtime.hpp"
#include "goals/goal.hpp"
#include "goals/no_goal.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace omni::hardware {

inline constexpr int GOAL_NONE = 0;
inline constexpr int GOAL_NADIR = 1;
inline constexpr int GOAL_ZENITH = 2;
inline constexpr int GOAL_ALONG_TRACK = 3;
inline constexpr int GOAL_CROSS_TRACK = 4;
inline constexpr int GOAL_VELOCITY = 5;
inline constexpr int GOAL_SUN = 6;
inline constexpr int GOAL_COORDINATE = 7;
inline constexpr int GOAL_COORDINATE_FMC = 8;
inline constexpr int GOAL_QUATERNION = 9;
inline constexpr int GOAL_ANGULAR_RATE = 10;

inline constexpr int GOAL_PARAM_WIDTH = 4;

class GoalTable {
public:
  GoalTable();
  int register_goal(const std::shared_ptr<goals::Goal> &goal);

  std::vector<int> kinds;
  std::vector<Eigen::Vector4d> params;
  std::vector<std::string> names;
  std::vector<std::shared_ptr<goals::Goal>> objects;
};

using DeviceMaskSpec = std::map<std::string, bool>;

std::vector<std::int8_t>
compile_sensor_mask(const DeviceMaskSpec &spec,
                    const std::vector<std::string> &sensor_names);

std::pair<std::vector<std::int8_t>, std::vector<std::int8_t>>
compile_actuator_masks(const DeviceMaskSpec &spec,
                       const std::vector<std::string> &rw_names,
                       const std::vector<std::string> &mtq_names);

void apply_hw_mask(const std::vector<std::int8_t> &mask,
                   std::vector<bool> &enabled);

struct ADCSMode {
  ADCSMode(std::string name,
           std::shared_ptr<goals::Goal> goal = std::make_shared<goals::NoGoal>(),
           std::shared_ptr<controllers::Controller> controller =
               std::make_shared<controllers::NoController>(),
           DeviceMaskSpec sensors = {}, DeviceMaskSpec actuators = {});

  std::string name;
  std::shared_ptr<goals::Goal> goal;
  std::shared_ptr<controllers::Controller> controller;
  DeviceMaskSpec sensors;
  DeviceMaskSpec actuators;
};

using ADCSTimelineEntry =
    std::pair<double, std::variant<std::string, std::shared_ptr<goals::Goal>>>;

struct ADCSStateMachine {
  ADCSStateMachine(std::string initial, std::vector<ADCSMode> modes,
                   std::vector<ADCSTimelineEntry> timeline = {});

  std::string initial;
  std::vector<ADCSMode> modes;
  std::vector<ADCSTimelineEntry> timeline;
};

struct ADCSSMRuntime {
  bool enabled = false;
  int initial_mode = 0;
  std::vector<int> mode_default_goal;
  std::vector<int> mode_default_controller;
  std::vector<std::vector<std::int8_t>> mode_sensor_mask;
  std::vector<std::vector<std::int8_t>> mode_rw_mask;
  std::vector<std::vector<std::int8_t>> mode_mtq_mask;
  std::vector<double> tl_times;
  std::vector<int> tl_modes;
  std::vector<int> tl_goals;
  std::vector<int> goal_kinds;
  std::vector<Eigen::Vector4d> goal_params;
  std::vector<std::shared_ptr<goals::Goal>> goal_objects;
  std::vector<std::string> mode_names;
  std::vector<std::string> goal_names;
};

ADCSSMRuntime prepare_adcs_sm(
    const ADCSStateMachine *adcs_sm, GoalTable *goal_table = nullptr,
    controllers::ControllerTable *controller_table = nullptr,
    const std::vector<std::string> &sensor_names = {},
    const std::vector<std::string> &rw_names = {},
    const std::vector<std::string> &mtq_names = {});

std::array<int, 3> init_adcs_work(const ADCSSMRuntime &runtime);

std::pair<int, int> step_adcs_sm(double elapsed_s, const ADCSSMRuntime &adcs_sm,
                                 std::array<int, 3> &work, int cmd_mode,
                                 int cmd_goal);

std::pair<Eigen::Vector4d, Eigen::Vector3d>
evaluate_goal(int goal_idx, const ADCSSMRuntime &adcs_sm,
              const Eigen::Ref<const Eigen::VectorXd> &state, double epoch_jd,
              const goals::GoalRuntime &runtime = {});

} // namespace omni::hardware

#endif

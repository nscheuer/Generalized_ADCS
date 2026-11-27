#ifndef ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_STATE_MACHINE_HPP
#define ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_STATE_MACHINE_HPP

#include "hardware/ADCS_subsystem/goals/goal.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace omni::hardware::obc {

using ADCSBinding = std::variant<
    std::string,
    std::shared_ptr<omni::hardware::goals::Goal>,
    std::pair<std::string, std::shared_ptr<omni::hardware::goals::Goal>>>;

using HardwareMaskSpec = std::map<std::string, bool>;

class Trigger {
public:
  virtual ~Trigger() = default;
};

class Telecommand final : public Trigger {
public:
  explicit Telecommand(
      std::string command,
      std::shared_ptr<omni::hardware::goals::Goal> goal = nullptr,
      std::optional<HardwareMaskSpec> sensors = std::nullopt,
      std::optional<HardwareMaskSpec> actuators = std::nullopt);

  const std::string &command() const { return command_; }
  const std::shared_ptr<omni::hardware::goals::Goal> &goal() const {
    return goal_;
  }
  const std::optional<HardwareMaskSpec> &sensors() const { return sensors_; }
  const std::optional<HardwareMaskSpec> &actuators() const {
    return actuators_;
  }

private:
  std::string command_;
  std::shared_ptr<omni::hardware::goals::Goal> goal_;
  std::optional<HardwareMaskSpec> sensors_;
  std::optional<HardwareMaskSpec> actuators_;
};

class InFOV final : public Trigger {
public:
  InFOV(double lat, double lon, double alt = 0.0,
        double max_off_nadir_deg = 45.0, bool negate = false,
        std::shared_ptr<omni::hardware::goals::Goal> goal = nullptr);

  InFOV inverted() const;

  double lat() const { return lat_; }
  double lon() const { return lon_; }
  double alt() const { return alt_; }
  double max_off_nadir_deg() const { return max_off_nadir_deg_; }
  bool negate() const { return negate_; }
  const std::shared_ptr<omni::hardware::goals::Goal> &goal() const {
    return goal_;
  }

private:
  double lat_;
  double lon_;
  double alt_;
  double max_off_nadir_deg_;
  bool negate_;
  std::shared_ptr<omni::hardware::goals::Goal> goal_;
};

class Completion final : public Trigger {};

class After final : public Trigger {
public:
  explicit After(double duration);
  double duration() const { return duration_; }

private:
  double duration_;
};

class FDIRTriggered final : public Trigger {
public:
  explicit FDIRTriggered(std::optional<std::string> monitor = std::nullopt)
      : monitor_(std::move(monitor)) {}
  const std::optional<std::string> &monitor() const { return monitor_; }

private:
  std::optional<std::string> monitor_;
};

struct ModeBindings {
  std::optional<ADCSBinding> adcs;
  std::optional<HardwareMaskSpec> sensors;
  std::optional<HardwareMaskSpec> actuators;
};

struct Mode {
  Mode(std::string name, std::vector<Mode> submodes = {},
       std::optional<std::string> initial_submode = std::nullopt,
       std::optional<double> duration = std::nullopt,
       std::optional<std::string> auto_target = std::nullopt,
       ModeBindings bindings = {});

  std::string name;
  std::vector<Mode> submodes;
  std::optional<std::string> initial_submode;
  std::optional<double> duration;
  std::optional<std::string> auto_target;
  ModeBindings bindings;
};

struct Transition {
  Transition(std::string source, std::string target,
             std::shared_ptr<Trigger> trigger, int priority = 0);

  std::string source;
  std::string target;
  std::shared_ptr<Trigger> trigger;
  int priority;
};

using TimelineEntry =
    std::tuple<double, std::string, std::shared_ptr<omni::hardware::goals::Goal>>;

struct StateMachine {
  StateMachine(std::string initial, std::vector<Mode> modes,
               std::vector<Transition> transitions = {},
               std::vector<TimelineEntry> timeline = {});

  std::string initial;
  std::vector<Mode> modes;
  std::vector<Transition> transitions;
  std::vector<TimelineEntry> timeline;
};

} // namespace omni::hardware::obc

#endif

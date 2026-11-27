#include "state_machine.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace omni::hardware::obc {

Telecommand::Telecommand(std::string command,
                         std::shared_ptr<omni::hardware::goals::Goal> goal,
                         std::optional<HardwareMaskSpec> sensors,
                         std::optional<HardwareMaskSpec> actuators)
    : command_(std::move(command)), goal_(std::move(goal)),
      sensors_(std::move(sensors)), actuators_(std::move(actuators)) {
  if (command_.empty()) {
    throw std::invalid_argument("telecommand name must be nonempty");
  }
}

InFOV::InFOV(double lat, double lon, double alt, double max_off_nadir_deg,
             bool negate, std::shared_ptr<omni::hardware::goals::Goal> goal)
    : lat_(lat), lon_(lon), alt_(alt), max_off_nadir_deg_(max_off_nadir_deg),
      negate_(negate), goal_(std::move(goal)) {
  if (!std::isfinite(lat_) || !std::isfinite(lon_) || !std::isfinite(alt_) ||
      !std::isfinite(max_off_nadir_deg_)) {
    throw std::invalid_argument("InFOV parameters must be finite");
  }
  if (lat_ < -90.0 || lat_ > 90.0) {
    throw std::invalid_argument("latitude must be between -90 and 90 degrees");
  }
  if (lon_ < -180.0 || lon_ > 180.0) {
    throw std::invalid_argument(
        "longitude must be between -180 and 180 degrees");
  }
  if (max_off_nadir_deg_ <= 0.0 || max_off_nadir_deg_ > 90.0) {
    throw std::invalid_argument("max_off_nadir_deg must be in (0, 90]");
  }
}

InFOV InFOV::inverted() const {
  return InFOV(lat_, lon_, alt_, max_off_nadir_deg_, !negate_, nullptr);
}

After::After(double duration) : duration_(duration) {
  if (!std::isfinite(duration_) || duration_ <= 0.0) {
    throw std::invalid_argument("duration must be finite and positive seconds");
  }
}

Mode::Mode(std::string name_val, std::vector<Mode> submodes_val,
           std::optional<std::string> initial_submode_val,
           std::optional<double> duration_val,
           std::optional<std::string> auto_target_val,
           ModeBindings bindings_val)
    : name(std::move(name_val)), submodes(std::move(submodes_val)),
      initial_submode(std::move(initial_submode_val)), duration(duration_val),
      auto_target(std::move(auto_target_val)),
      bindings(std::move(bindings_val)) {
  if (name.empty() || name.find('.') != std::string::npos ||
      name.find('*') != std::string::npos) {
    throw std::invalid_argument("invalid mode name: " + name);
  }
  if (duration.has_value() &&
      (!std::isfinite(*duration) || *duration <= 0.0)) {
    throw std::invalid_argument(
        "mode duration must be finite and positive seconds");
  }
  if (!submodes.empty()) {
    std::set<std::string> seen;
    for (const auto &sub : submodes) {
      if (!seen.insert(sub.name).second) {
        throw std::invalid_argument("duplicate submode names in " + name);
      }
    }
    if (!initial_submode.has_value()) {
      initial_submode = submodes.front().name;
    } else if (seen.find(*initial_submode) == seen.end()) {
      throw std::invalid_argument("initial_submode " + *initial_submode +
                                  " not in " + name);
    }
  } else if (initial_submode.has_value()) {
    throw std::invalid_argument("initial_submode requires submodes");
  }
}

Transition::Transition(std::string source_val, std::string target_val,
                       std::shared_ptr<Trigger> trigger_val, int priority_val)
    : source(std::move(source_val)), target(std::move(target_val)),
      trigger(std::move(trigger_val)), priority(priority_val) {
  if (source.empty() || target.empty()) {
    throw std::invalid_argument(
        "transition source and target must be nonempty");
  }
  if (!trigger) {
    throw std::invalid_argument("invalid trigger: null");
  }
}

StateMachine::StateMachine(std::string initial_val, std::vector<Mode> modes_val,
                           std::vector<Transition> transitions_val,
                           std::vector<TimelineEntry> timeline_val)
    : initial(std::move(initial_val)), modes(std::move(modes_val)),
      transitions(std::move(transitions_val)),
      timeline(std::move(timeline_val)) {
  if (modes.empty()) {
    throw std::invalid_argument("StateMachine requires at least one Mode");
  }
  std::set<std::string> seen;
  for (const auto &m : modes) {
    if (!seen.insert(m.name).second) {
      throw std::invalid_argument("duplicate top-level mode names");
    }
  }
  if (initial.empty()) {
    throw std::invalid_argument("initial mode must be nonempty");
  }
  for (const auto &entry : timeline) {
    const double t = std::get<0>(entry);
    const auto &cmd = std::get<1>(entry);
    if (!std::isfinite(t) || t < 0.0) {
      throw std::invalid_argument(
          "timeline times must be finite and nonnegative");
    }
    if (cmd.empty()) {
      throw std::invalid_argument(
          "timeline telecommand names must be nonempty");
    }
  }
}

} // namespace omni::hardware::obc

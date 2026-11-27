#include "obc_subsystem.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::hardware::obc {

OBCSubsystem::OBCSubsystem(std::shared_ptr<StateMachine> state_machine_val,
                           std::shared_ptr<FDIR> fdir_val, int N_val,
                           std::vector<TimelineEntry> timeline_val)
    : state_machine(std::move(state_machine_val)), fdir(std::move(fdir_val)),
      N(N_val), timeline(std::move(timeline_val)) {
  if (N < 1) {
    throw std::invalid_argument("N must be a positive integer");
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

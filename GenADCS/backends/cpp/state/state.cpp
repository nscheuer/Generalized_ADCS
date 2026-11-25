#include "state.hpp"

#include <stdexcept>
#include <string>

namespace omni {
State::State(const AttitudeState &attitude_value,
             const OrbitalState &orbit_value,
             const InternalState &internal_value)
    : attitude(attitude_value), orbit(orbit_value), internal(internal_value) {}

void State::check_wheel_count(const std::optional<std::size_t> &count) const {
  if (count.has_value() &&
      (count.value() !=
       static_cast<std::size_t>(internal.wheel_momentum.size()))) {
    throw std::invalid_argument("expected " + std::to_string(count.value()) +
                                " reaction wheel momenta, got " +
                                std::to_string(internal.wheel_momentum.size()));
  }
}
} // namespace omni

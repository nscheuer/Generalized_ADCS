#ifndef ADCS_OMNI_STATE_HPP
#define ADCS_OMNI_STATE_HPP

#include "attitude_state.hpp"
#include "internal_state.hpp"
#include "orbital_state.hpp"

#include <cstddef>
#include <optional>

namespace omni {
struct State {
  AttitudeState attitude;
  OrbitalState orbit;
  InternalState internal;

  State(const AttitudeState &attitude_value, const OrbitalState &orbit_value,
        const InternalState &internal_value = InternalState{});
  void check_wheel_count(const std::optional<std::size_t> &count) const;
};
} // namespace omni

#endif // ADCS_OMNI_STATE_HPP

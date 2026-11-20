#include "orbital_state.hpp"

#include "state_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace omni {
OrbitalState::OrbitalState(const std::optional<Vector3>& position_value,
                           const std::optional<Vector3>& velocity_value,
                           double julian_date_value,
                           const std::optional<KeplerianElements>& elements)
    : position(position_value),
      velocity(velocity_value),
      julian_date(julian_date_value),
      keplerian_elements(elements) {
    if (!std::isfinite(julian_date)) {
        throw std::invalid_argument("julian_date must be finite");
    }
    if (position.has_value()) {
        detail::require_finite(position.value(), "position");
    }
    if (velocity.has_value()) {
        detail::require_finite(velocity.value(), "velocity");
    }
}

Vector3 OrbitalState::cartesian_position() const {
    if (!position.has_value() || !velocity.has_value()) {
        throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    }
    return position.value();
}

Vector3 OrbitalState::cartesian_velocity() const {
    if (!position.has_value() || !velocity.has_value()) {
        throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    }
    return velocity.value();
}
}

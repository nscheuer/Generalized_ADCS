#include "state.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace omni {
namespace {
template <std::size_t N>
void check_finite(const std::array<double, N>& value, const char* name) {
    for (double component : value) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }
}
}

AttitudeState::AttitudeState(Quaternion orientation_, Vector3 rates_)
    : orientation(orientation_), rates(rates_) {
    check_finite(orientation, "orientation");
    check_finite(rates, "rates");
    double norm_squared = 0.0;
    for (double component : orientation) norm_squared += component * component;
    if (norm_squared == 0.0 || !std::isfinite(norm_squared)) {
        throw std::invalid_argument("orientation quaternion must be finite and nonzero");
    }
}

Quaternion AttitudeState::quaternion_array() const {
    double norm_squared = 0.0;
    for (double component : orientation) norm_squared += component * component;
    if (norm_squared == 0.0 || !std::isfinite(norm_squared)) {
        throw std::invalid_argument("orientation quaternion must be finite and nonzero");
    }
    auto result = orientation;
    const double norm = std::sqrt(norm_squared);
    for (double& component : result) component /= norm;
    return result;
}

InternalState::InternalState(std::vector<double> momentum) : wheel_momentum(std::move(momentum)) {
    for (double component : wheel_momentum) {
        if (!std::isfinite(component)) throw std::invalid_argument("wheel_momentum must be finite");
    }
}

OrbitalState::OrbitalState(std::optional<Vector3> position_, std::optional<Vector3> velocity_,
                           double julian_date_, std::optional<std::array<double, 6>> elements)
    : position(position_), velocity(velocity_), julian_date(julian_date_), keplerian_elements(elements) {
    if (!std::isfinite(julian_date)) throw std::invalid_argument("julian_date must be finite");
    if (position) check_finite(*position, "position");
    if (velocity) check_finite(*velocity, "velocity");
}

Vector3 OrbitalState::cartesian_position() const {
    if (!position || !velocity) throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    return *position;
}

Vector3 OrbitalState::cartesian_velocity() const {
    if (!position || !velocity) throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    return *velocity;
}

State::State(AttitudeState attitude_, OrbitalState orbit_, InternalState internal_)
    : attitude(std::move(attitude_)), orbit(std::move(orbit_)), internal(std::move(internal_)) {}

void State::check_wheel_count(std::optional<std::size_t> count) const {
    if (count && *count != internal.wheel_momentum.size()) {
        throw std::invalid_argument("expected " + std::to_string(*count) +
                                    " reaction wheel momenta, got " +
                                    std::to_string(internal.wheel_momentum.size()));
    }
}
}

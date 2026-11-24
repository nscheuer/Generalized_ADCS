#include "orbital_state.hpp"

#include "state_validation.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni
{
OrbitalState::OrbitalState(const std::optional<Vector3> &position_value, const std::optional<Vector3> &velocity_value,
                           double julian_date_value, const std::optional<KeplerianElements> &elements,
                           std::shared_ptr<const EphemerisProvider> ephemeris_value)
    : position(position_value), velocity(velocity_value), julian_date(julian_date_value),
      ephemeris(std::move(ephemeris_value)), keplerian_elements(elements)
{
    if (!std::isfinite(julian_date))
    {
        throw std::invalid_argument("julian_date must be finite");
    }
    if (position.has_value())
    {
        detail::require_finite(position.value(), "position");
    }
    if (velocity.has_value())
    {
        detail::require_finite(velocity.value(), "velocity");
    }
}

Vector3 OrbitalState::cartesian_position() const
{
    if (!position.has_value() || !velocity.has_value())
    {
        throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    }
    return position.value();
}

Vector3 OrbitalState::cartesian_velocity() const
{
    if (!position.has_value() || !velocity.has_value())
    {
        throw std::logic_error("Keplerian to Cartesian conversion is not implemented");
    }
    return velocity.value();
}

Vector3 OrbitalState::body_position(const std::string &cache_key, int body_id)
{
    const auto cached = body_positions.find(cache_key);
    if (cached != body_positions.end())
        return cached->second;
    if (!ephemeris)
        throw std::logic_error("an ephemeris must be configured");
    Vector3 result = ephemeris->position(body_id, julian_date);
    detail::require_finite(result, "body position");
    body_positions.emplace(cache_key, result);
    return result;
}
} // namespace omni

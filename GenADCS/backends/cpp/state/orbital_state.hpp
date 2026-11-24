#ifndef ADCS_OMNI_ORBITAL_STATE_HPP
#define ADCS_OMNI_ORBITAL_STATE_HPP

#include "state_types.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace omni
{
class EphemerisProvider
{
  public:
    virtual ~EphemerisProvider() = default;
    virtual Vector3 position(int body_id, double julian_date) const = 0;
};

struct OrbitalState
{
    std::optional<Vector3> position;
    std::optional<Vector3> velocity;
    double julian_date;
    std::shared_ptr<const EphemerisProvider> ephemeris;
    std::map<std::string, Vector3> body_positions;
    std::optional<KeplerianElements> keplerian_elements;

    OrbitalState(const std::optional<Vector3> &position_value, const std::optional<Vector3> &velocity_value,
                 double julian_date_value, const std::optional<KeplerianElements> &elements = std::nullopt,
                 std::shared_ptr<const EphemerisProvider> ephemeris_value = nullptr);
    Vector3 cartesian_position() const;
    Vector3 cartesian_velocity() const;
    Vector3 body_position(const std::string &cache_key, int body_id);
};
} // namespace omni

#endif // ADCS_OMNI_ORBITAL_STATE_HPP

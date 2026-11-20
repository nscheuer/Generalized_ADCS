#ifndef ADCS_OMNI_ORBITAL_STATE_HPP
#define ADCS_OMNI_ORBITAL_STATE_HPP

#include "state_types.hpp"

#include <map>
#include <optional>
#include <string>

#include <pybind11/pybind11.h>

namespace omni {
struct OrbitalState {
    std::optional<Vector3> position;
    std::optional<Vector3> velocity;
    double julian_date;
    pybind11::object ephemeris = pybind11::none();
    std::map<std::string, pybind11::object> body_positions;
    std::optional<KeplerianElements> keplerian_elements;

    OrbitalState(const std::optional<Vector3>& position_value,
                 const std::optional<Vector3>& velocity_value,
                 double julian_date_value,
                 const std::optional<KeplerianElements>& elements = std::nullopt);
    Vector3 cartesian_position() const;
    Vector3 cartesian_velocity() const;
};
}

#endif  // ADCS_OMNI_ORBITAL_STATE_HPP

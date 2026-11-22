#include "satellite_geometry.hpp"

#include <utility>

namespace omni::hardware {
SatelliteGeometry::SatelliteGeometry(double mass_value,
                                     pybind11::object inertia_value,
                                     pybind11::list panels_value,
                                     pybind11::object center_of_mass_value)
    : mass(mass_value),
      inertia(std::move(inertia_value)),
      panels(std::move(panels_value)),
      center_of_mass(std::move(center_of_mass_value)) {}
}

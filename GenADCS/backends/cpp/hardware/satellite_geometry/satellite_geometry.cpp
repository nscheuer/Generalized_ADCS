#include "satellite_geometry.hpp"

#include <utility>

namespace omni::hardware
{
SatelliteGeometry::SatelliteGeometry(double mass_value, std::array<double, 9> inertia_value,
                                     std::vector<SatellitePanel> panels_value,
                                     std::array<double, 3> center_of_mass_value)
    : mass(mass_value), inertia(std::move(inertia_value)), panels(std::move(panels_value)),
      center_of_mass(std::move(center_of_mass_value))
{
}
} // namespace omni::hardware

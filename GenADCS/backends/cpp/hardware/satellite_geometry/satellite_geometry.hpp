#ifndef ADCS_OMNI_HARDWARE_SATELLITE_GEOMETRY_HPP
#define ADCS_OMNI_HARDWARE_SATELLITE_GEOMETRY_HPP

#include "satellite_panel.hpp"

#include <array>
#include <vector>

namespace omni::hardware
{
struct SatelliteGeometry
{
    SatelliteGeometry(double mass_value, std::array<double, 9> inertia_value,
                      std::vector<SatellitePanel> panels_value = {},
                      std::array<double, 3> center_of_mass_value = {0.0, 0.0, 0.0});

    double mass;
    std::array<double, 9> inertia;
    std::vector<SatellitePanel> panels;
    std::array<double, 3> center_of_mass;
};
} // namespace omni::hardware

#endif

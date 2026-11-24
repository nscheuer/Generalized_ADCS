#pragma once

#include "satellite_geometry/satellite_geometry.hpp"

#include <string>

namespace omni::hardware
{

// Native satellite configuration. Python-facing subsystem objects are handled
// by the binding layer; this type owns only native C++ values.
struct Satellite
{
    explicit Satellite(SatelliteGeometry geometry_value, std::string initial_mode = "idle");

    SatelliteGeometry geometry;
    std::string mode;
};

} // namespace omni::hardware

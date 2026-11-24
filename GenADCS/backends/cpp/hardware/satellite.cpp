#include "satellite.hpp"

#include <utility>

namespace omni::hardware
{

Satellite::Satellite(SatelliteGeometry geometry_value, std::string initial_mode)
    : geometry(std::move(geometry_value)), mode(std::move(initial_mode))
{
}

} // namespace omni::hardware

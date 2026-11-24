#include "magnetorquer.hpp"

#include <utility>

namespace omni::hardware
{
Magnetorquer::Magnetorquer(V position_value, V orientation_value, double max_dipole_value, V misalignment_value)
    : position(std::move(position_value)), orientation(std::move(orientation_value)), max_dipole(max_dipole_value),
      misalignment(std::move(misalignment_value))
{
}
} // namespace omni::hardware

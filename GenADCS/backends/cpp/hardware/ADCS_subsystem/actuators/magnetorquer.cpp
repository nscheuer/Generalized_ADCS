#include "magnetorquer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::hardware {
Magnetorquer::Magnetorquer(V position_value, V orientation_value,
                           double max_dipole_value, V misalignment_value,
                           double noise_value, double bias_value)
    : position(std::move(position_value)),
      orientation(std::move(orientation_value)), max_dipole(max_dipole_value),
      misalignment(std::move(misalignment_value)), noise(noise_value),
      bias(bias_value) {
  if (!std::isfinite(noise) || noise < 0.0) {
    throw std::invalid_argument("noise must be finite and nonnegative");
  }
  if (!std::isfinite(bias)) {
    throw std::invalid_argument("bias must be finite");
  }
}
} // namespace omni::hardware

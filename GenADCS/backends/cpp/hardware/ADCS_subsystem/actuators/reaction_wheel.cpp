#include "reaction_wheel.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware {
ReactionWheel::ReactionWheel(Vector3 position_value, Vector3 orientation_value,
                             double rotational_inertia_value,
                             double max_torque_value, double max_momentum_value,
                             Vector3 misalignment_value, double noise_value,
                             double bias_value)
    : position(position_value), orientation(orientation_value),
      rotational_inertia(rotational_inertia_value),
      max_torque(max_torque_value), max_momentum(max_momentum_value),
      misalignment(misalignment_value), noise(noise_value), bias(bias_value) {
  if (!std::isfinite(noise) || noise < 0.0) {
    throw std::invalid_argument("noise must be finite and nonnegative");
  }
  if (!std::isfinite(bias)) {
    throw std::invalid_argument("bias must be finite");
  }
}
} // namespace omni::hardware

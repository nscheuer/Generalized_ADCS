#include "reaction_wheel.hpp"

namespace omni::hardware {
ReactionWheel::ReactionWheel(Vector3 position_value, Vector3 orientation_value,
                             double rotational_inertia_value,
                             double max_torque_value, double max_momentum_value,
                             Vector3 misalignment_value)
    : position(position_value), orientation(orientation_value),
      rotational_inertia(rotational_inertia_value),
      max_torque(max_torque_value), max_momentum(max_momentum_value),
      misalignment(misalignment_value) {}
} // namespace omni::hardware

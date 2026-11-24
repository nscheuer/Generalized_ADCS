#ifndef ADCS_OMNI_HARDWARE_REACTION_WHEEL_HPP
#define ADCS_OMNI_HARDWARE_REACTION_WHEEL_HPP

#include <array>

namespace omni::hardware
{
struct ReactionWheel
{
    using Vector3 = std::array<double, 3>;

    ReactionWheel(Vector3 position_value, Vector3 orientation_value, double rotational_inertia_value,
                  double max_torque_value, double max_momentum_value, Vector3 misalignment_value = {{0.0, 0.0, 0.0}});

    Vector3 position;
    Vector3 orientation;
    double rotational_inertia;
    double max_torque;
    double max_momentum;
    Vector3 misalignment;
};
} // namespace omni::hardware

#endif

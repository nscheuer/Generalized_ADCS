#include "reaction_wheel.hpp"
#include "effects.hpp"

#include <utility>

namespace omni::hardware {
ReactionWheel::ReactionWheel(pybind11::object position_value,
                             pybind11::object orientation_value,
                             double rotational_inertia_value,
                             double max_torque_value,
                             double max_momentum_value,
                             pybind11::object misalignment_value,
                             pybind11::list effects_value)
    : position(std::move(position_value)),
      orientation(std::move(orientation_value)),
      rotational_inertia(rotational_inertia_value),
      max_torque(max_torque_value),
      max_momentum(max_momentum_value),
      misalignment(std::move(misalignment_value)),
      effects(std::move(effects_value)) {}

pybind11::list ReactionWheel::noises() const {
    return filter_effects(effects, "ADCS_Omni.ADCS_Python.noise", "Noise");
}
pybind11::list ReactionWheel::biases() const {
    return filter_effects(effects, "ADCS_Omni.ADCS_Python.bias", "Bias");
}
pybind11::object ReactionWheel::apply_errors(pybind11::object value, double dt,
                                             pybind11::object rng) const {
    return apply_effects(std::move(value), effects, dt, std::move(rng));
}
}

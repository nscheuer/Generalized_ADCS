#include "magnetorquer.hpp"
#include "effects.hpp"

#include <utility>

namespace omni::hardware {
Magnetorquer::Magnetorquer(pybind11::object position_value,
                           pybind11::object orientation_value,
                           double max_dipole_value,
                           pybind11::object misalignment_value,
                           pybind11::list effects_value)
    : position(std::move(position_value)),
      orientation(std::move(orientation_value)),
      max_dipole(max_dipole_value),
      misalignment(std::move(misalignment_value)),
      effects(std::move(effects_value)) {}

pybind11::list Magnetorquer::noises() const {
    return filter_effects(effects, "GenADCS.backends.python.noise", "Noise");
}
pybind11::list Magnetorquer::biases() const {
    return filter_effects(effects, "GenADCS.backends.python.bias", "Bias");
}
pybind11::object Magnetorquer::apply_errors(pybind11::object value, double dt,
                                            pybind11::object rng) const {
    return apply_effects(std::move(value), effects, dt, std::move(rng));
}
}

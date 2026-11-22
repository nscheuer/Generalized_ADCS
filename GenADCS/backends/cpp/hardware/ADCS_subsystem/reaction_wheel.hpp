#ifndef ADCS_OMNI_HARDWARE_REACTION_WHEEL_HPP
#define ADCS_OMNI_HARDWARE_REACTION_WHEEL_HPP

#include <pybind11/pybind11.h>

namespace omni::hardware {
struct ReactionWheel {
    ReactionWheel(pybind11::object position_value,
                  pybind11::object orientation_value,
                  double rotational_inertia_value,
                  double max_torque_value,
                  double max_momentum_value,
                  pybind11::object misalignment_value,
                  pybind11::list effects_value);
    pybind11::list noises() const;
    pybind11::list biases() const;
    pybind11::object apply_errors(pybind11::object value, double dt,
                                  pybind11::object rng) const;

    pybind11::object position;
    pybind11::object orientation;
    double rotational_inertia;
    double max_torque;
    double max_momentum;
    pybind11::object misalignment;
    pybind11::list effects;
};
}

#endif

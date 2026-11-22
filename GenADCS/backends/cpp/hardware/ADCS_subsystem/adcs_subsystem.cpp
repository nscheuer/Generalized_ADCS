#include "adcs_subsystem.hpp"

#include <utility>

namespace omni::hardware {
ADCSSubsystem::ADCSSubsystem(pybind11::list reaction_wheels_value,
                             pybind11::list magnetorquers_value)
    : reaction_wheels(std::move(reaction_wheels_value)),
      magnetorquers(std::move(magnetorquers_value)) {}
}

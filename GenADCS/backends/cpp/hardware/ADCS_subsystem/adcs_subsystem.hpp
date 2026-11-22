#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_HPP

#include <pybind11/pybind11.h>

namespace omni::hardware {
struct ADCSSubsystem {
    ADCSSubsystem(pybind11::list reaction_wheels_value,
                  pybind11::list magnetorquers_value);
    pybind11::list reaction_wheels;
    pybind11::list magnetorquers;
};
}

#endif

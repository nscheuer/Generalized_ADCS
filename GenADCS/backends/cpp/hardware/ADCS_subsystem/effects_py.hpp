#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_EFFECTS_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_EFFECTS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace omni::hardware {
pybind11::object apply_effects(pybind11::object value,
                               const pybind11::sequence &effects, double dt,
                               pybind11::object rng = pybind11::none());
pybind11::list filter_effects(const pybind11::sequence &effects,
                              const char *module_name, const char *class_name);
} // namespace omni::hardware

#endif

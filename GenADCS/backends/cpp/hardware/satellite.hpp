#pragma once
#include <pybind11/pybind11.h>
namespace omni::hardware {
struct Satellite {
    pybind11::object geometry;
    pybind11::object adcs;
    pybind11::object state_machine;
    Satellite(pybind11::object geometry_value, pybind11::object adcs_value,
              pybind11::object state_machine_value);
};
}

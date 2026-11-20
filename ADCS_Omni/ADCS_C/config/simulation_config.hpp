#pragma once
#include <pybind11/pybind11.h>
namespace omni::config {
struct SimulationConfig { pybind11::object satellite,environment,state,propagator,solver; pybind11::object backend; pybind11::object state_record; double julian_date; SimulationConfig(pybind11::object,pybind11::object,pybind11::object,pybind11::object,pybind11::object,pybind11::object); };
}

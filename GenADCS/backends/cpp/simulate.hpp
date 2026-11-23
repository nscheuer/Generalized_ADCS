#pragma once
#include <pybind11/pybind11.h>
#include "helpers/input_output/resultsformat.hpp"
namespace omni::simulation {
namespace py = pybind11;
py::object simulate(py::object satellite, py::object environment,
                          py::object dynamics, py::object state,
                          py::object solver, const std::string& backend,
                          py::object resultsformat);
}

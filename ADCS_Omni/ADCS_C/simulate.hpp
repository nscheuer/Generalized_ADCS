#pragma once
#include <pybind11/pybind11.h>
namespace omni::simulation {
namespace py = pybind11;
struct SimulationResult {
    py::object times_s;
    py::object state_vectors;
    py::object final_state;
};
SimulationResult simulate(py::object satellite, py::object environment,
                          py::object dynamics, py::object state,
                          py::object solver, const std::string& backend);
}

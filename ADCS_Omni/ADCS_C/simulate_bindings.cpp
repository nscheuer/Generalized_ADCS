#include "simulate.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_simulate, module) {
    py::class_<omni::simulation::SimulationResult>(module, "SimulationResult")
        .def_readonly("times_s", &omni::simulation::SimulationResult::times_s)
        .def_readonly("state_vectors", &omni::simulation::SimulationResult::state_vectors)
        .def_readonly("final_state", &omni::simulation::SimulationResult::final_state);
    module.def("simulate", &omni::simulation::simulate,
        py::arg("satellite"), py::arg("environment"), py::arg("dynamics"),
        py::arg("state"), py::arg("solver"), py::arg("backend") = "C");
}

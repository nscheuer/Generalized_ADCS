#include "simulate.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_simulate, module) {
    module.attr("ResultsFormat") = py::module_::import(
        "GenADCS.backends.shared.simulationresult.resultsformat").attr("ResultsFormat");
    module.def("simulate", &omni::simulation::simulate,
        py::arg("satellite"), py::arg("environment"), py::arg("dynamics"),
        py::arg("state"), py::arg("solver"), py::arg("backend") = "C",
        py::arg("resultsformat") = py::none());
}

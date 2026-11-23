#include "simulate.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_simulate, module) {
    py::class_<omni::simulation::ResultsFormat>(module, "ResultsFormat")
        .def(py::init([](bool attitude_quaternion, bool satellite_rate, bool wheel_momenta, std::size_t N) {
            if (!N) throw py::value_error("N must be a positive integer");
            return omni::simulation::ResultsFormat{attitude_quaternion, satellite_rate, wheel_momenta, N};
        }), py::arg("attitude_quaternion") = true, py::arg("satellite_rate") = true,
            py::arg("wheel_momenta") = true, py::arg("N") = 1)
        .def_readonly("attitude_quaternion", &omni::simulation::ResultsFormat::attitude_quaternion)
        .def_readonly("satellite_rate", &omni::simulation::ResultsFormat::satellite_rate)
        .def_readonly("wheel_momenta", &omni::simulation::ResultsFormat::wheel_momenta)
        .def_readonly("N", &omni::simulation::ResultsFormat::N);
    module.def("simulate", &omni::simulation::simulate,
        py::arg("satellite"), py::arg("environment"), py::arg("dynamics"),
        py::arg("state"), py::arg("solver"), py::arg("backend") = "C",
        py::arg("resultsformat") = py::none());
}

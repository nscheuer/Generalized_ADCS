#include "satellite.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_satellite, module) {
    py::class_<omni::hardware::Satellite>(module, "Satellite")
        .def(py::init<py::object, py::object, py::object>(),
             py::arg("geometry"), py::arg("adcs") = py::none(),
             py::arg("state_machine") = py::none())
        .def_readwrite("geometry", &omni::hardware::Satellite::geometry)
        .def_readwrite("adcs", &omni::hardware::Satellite::adcs)
        .def_readwrite("state_machine", &omni::hardware::Satellite::state_machine);
}

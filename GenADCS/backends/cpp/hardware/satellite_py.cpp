#include "satellite.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_satellite, module) {
    // Ensure the SatelliteGeometry caster is registered before exposing Satellite.
    py::module_::import("GenADCS.backends.cpp.hardware.satellite_geometry");

    py::class_<omni::hardware::Satellite>(module, "Satellite", py::dynamic_attr())
        .def(py::init<omni::hardware::SatelliteGeometry, std::string>(),
             py::arg("geometry"), py::arg("initial_mode") = "idle")
        .def_readwrite("geometry", &omni::hardware::Satellite::geometry)
        .def_readwrite("mode", &omni::hardware::Satellite::mode);
}

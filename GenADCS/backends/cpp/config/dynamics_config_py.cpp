#include "dynamics_config.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
using PythonDynamicsConfig = omni::config::BasicDynamicsConfig<py::object>;
}

void bind_dynamics_config(py::module_& m) {
    py::class_<PythonDynamicsConfig>(m, "DynamicsConfig")
        .def(py::init<>())
        .def(py::init<std::vector<py::object>, std::vector<py::object>>(),
             py::arg("forces") = std::vector<py::object>{},
             py::arg("torques") = std::vector<py::object>{})
        .def_readwrite("forces", &PythonDynamicsConfig::forces)
        .def_readwrite("torques", &PythonDynamicsConfig::torques);
}

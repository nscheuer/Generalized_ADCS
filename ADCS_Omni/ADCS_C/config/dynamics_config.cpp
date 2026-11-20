#include "dynamics_config.hpp"
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_dynamics_config(py::module_& m) {
 py::class_<omni::config::DynamicsConfig>(m,"DynamicsConfig")
 .def(py::init<>()).def(py::init<std::vector<py::object>,std::vector<py::object>>(),py::arg("forces")=std::vector<py::object>{},py::arg("torques")=std::vector<py::object>{})
 .def_readwrite("forces",&omni::config::DynamicsConfig::forces).def_readwrite("torques",&omni::config::DynamicsConfig::torques);
}

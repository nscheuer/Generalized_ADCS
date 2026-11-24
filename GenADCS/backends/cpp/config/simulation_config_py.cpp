#include "simulation_config.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_simulation_config(py::module_& m){
 py::class_<omni::config::SimulationConfig>(m,"SimulationConfig").def(py::init<py::object,py::object,py::object,py::object,py::object,py::object>(),py::arg("satellite"),py::arg("environment"),py::arg("state"),py::arg("propagator"),py::arg("solver"),py::arg("backend")="Python")
 .def_readwrite("satellite",&omni::config::SimulationConfig::satellite).def_readwrite("environment",&omni::config::SimulationConfig::environment).def_readwrite("state",&omni::config::SimulationConfig::state).def_readwrite("propagator",&omni::config::SimulationConfig::propagator).def_readwrite("solver",&omni::config::SimulationConfig::solver).def_readwrite("backend",&omni::config::SimulationConfig::backend).def_readonly("state_record",&omni::config::SimulationConfig::state_record).def_readonly("julian_date",&omni::config::SimulationConfig::julian_date);
}

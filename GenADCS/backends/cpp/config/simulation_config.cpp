#include "simulation_config.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "dynamics_config.hpp"
#include <stdexcept>
namespace py=pybind11;
namespace omni::config {
SimulationConfig::SimulationConfig(py::object sat,py::object env,py::object st,py::object prop,py::object sol,py::object be):satellite(sat),environment(env),state(st),propagator(prop),solver(sol),backend(be){
 std::string name=py::str(backend);if(name!="Python"&&name!="C"&&name!="Rust"&&name!="JAX")throw py::value_error("backend must be Python, C, Rust, or JAX");
 state_record=state.attr("to_record")(py::len(satellite.attr("adcs").attr("reaction_wheels")));
 julian_date=state.attr("orbit").attr("julian_date").cast<double>();
}
}
void bind_simulation_config(py::module_& m){
 py::class_<omni::config::SimulationConfig>(m,"SimulationConfig").def(py::init<py::object,py::object,py::object,py::object,py::object,py::object>(),py::arg("satellite"),py::arg("environment"),py::arg("state"),py::arg("propagator"),py::arg("solver"),py::arg("backend")="Python")
 .def_readwrite("satellite",&omni::config::SimulationConfig::satellite).def_readwrite("environment",&omni::config::SimulationConfig::environment).def_readwrite("state",&omni::config::SimulationConfig::state).def_readwrite("propagator",&omni::config::SimulationConfig::propagator).def_readwrite("solver",&omni::config::SimulationConfig::solver).def_readwrite("backend",&omni::config::SimulationConfig::backend).def_readonly("state_record",&omni::config::SimulationConfig::state_record).def_readonly("julian_date",&omni::config::SimulationConfig::julian_date);
}

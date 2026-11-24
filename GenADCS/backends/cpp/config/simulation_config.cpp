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

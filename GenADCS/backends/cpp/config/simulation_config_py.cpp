#include "simulation_config.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace py = pybind11;

namespace {

struct PythonSimulationConfig {
  py::object satellite;
  py::object environment;
  py::object state;
  py::object propagator;
  py::object solver;
  py::object backend;
  py::object state_record;
  double julian_date;

  PythonSimulationConfig(py::object satellite_value,
                         py::object environment_value, py::object state_value,
                         py::object propagator_value, py::object solver_value,
                         py::object backend_value)
      : satellite(std::move(satellite_value)),
        environment(std::move(environment_value)),
        state(std::move(state_value)), propagator(std::move(propagator_value)),
        solver(std::move(solver_value)), backend(std::move(backend_value)),
        julian_date(0.0) {
    const omni::config::SimulationConfig native_config(
        py::str(backend).cast<std::string>());
    state_record = state.attr("to_record")(
        py::len(satellite.attr("adcs").attr("reaction_wheels")));
    julian_date = state.attr("orbit").attr("julian_date").cast<double>();
  }
};

} // namespace

void bind_simulation_config(py::module_ &module) {
  py::class_<PythonSimulationConfig>(module, "SimulationConfig")
      .def(py::init<py::object, py::object, py::object, py::object, py::object,
                    py::object>(),
           py::arg("satellite"), py::arg("environment"), py::arg("state"),
           py::arg("propagator"), py::arg("solver"),
           py::arg("backend") = "Python")
      .def_readwrite("satellite", &PythonSimulationConfig::satellite)
      .def_readwrite("environment", &PythonSimulationConfig::environment)
      .def_readwrite("state", &PythonSimulationConfig::state)
      .def_readwrite("propagator", &PythonSimulationConfig::propagator)
      .def_readwrite("solver", &PythonSimulationConfig::solver)
      .def_readwrite("backend", &PythonSimulationConfig::backend)
      .def_readonly("state_record", &PythonSimulationConfig::state_record)
      .def_readonly("julian_date", &PythonSimulationConfig::julian_date);
}

#include "attitude_solver.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_attitude_solver(py::module_ &m) {
  py::class_<omni::solver::AttitudeSolver>(m, "_AttitudeSolver")
      .def(py::init<double>(), py::arg("dt"))
      .def_property_readonly("dt", &omni::solver::AttitudeSolver::dt);
}

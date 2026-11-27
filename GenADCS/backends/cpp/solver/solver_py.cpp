#include "coupled_solver.hpp"
#include "solver.hpp"
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace {
using PythonSolver =
    omni::solver::BasicSolver<py::object, py::object, py::object, py::object>;

std::optional<py::object> optional_object(const py::object &value) {
  if (value.is_none()) {
    return std::nullopt;
  }
  return value;
}

py::object object_or_none(const std::optional<py::object> &value) {
  return value.has_value() ? *value : py::none();
}

std::optional<double> optional_time(const py::object &value) {
  if (value.is_none()) {
    return std::nullopt;
  }
  return py::module_::import("builtins").attr("float")(value).cast<double>();
}
} // namespace

void bind_solver(py::module_ &m) {
  py::class_<PythonSolver>(m, "Solver")
      .def(py::init([](py::object dynamics, py::object orbital,
                       py::object attitude, py::object coupled, py::object tf,
                       bool progress, long long progress_chunk_steps) {
             if (progress_chunk_steps < 1 ||
                 progress_chunk_steps > std::numeric_limits<int>::max()) {
               throw py::value_error(
                   "progress_chunk_steps must be a positive integer");
             }
             if (dynamics.is_none()) {
               dynamics = py::module_::import("GenADCS.backends.cpp.config")
                              .attr("DynamicsConfig")();
             }
             return PythonSolver(std::move(dynamics), optional_object(orbital),
                                 optional_object(attitude), optional_object(coupled),
                                 optional_time(tf), progress,
                                 static_cast<int>(progress_chunk_steps));
           }),
           py::arg("dynamics") = py::none(), py::arg("orbital") = py::none(),
           py::arg("attitude") = py::none(), py::arg("coupled") = py::none(),
           py::arg("tf") = py::none(), py::arg("progress") = true,
           py::arg("progress_chunk_steps") = 500)
      .def_readwrite("dynamics", &PythonSolver::dynamics)
      .def_readwrite("progress", &PythonSolver::progress)
      .def_readwrite("progress_chunk_steps", &PythonSolver::progress_chunk_steps)
      .def_property(
          "tf", [](const PythonSolver &self) -> py::object {
            if (self.tf.has_value()) {
              return py::float_(*self.tf);
            }
            return py::none();
          },
          [](PythonSolver &self, py::object value) {
            self.tf = optional_time(value);
          })
      .def_property(
          "orbital",
          [](const PythonSolver &self) { return object_or_none(self.orbital); },
          [](PythonSolver &self, py::object value) {
            self.orbital = optional_object(value);
          })
      .def_property(
          "attitude",
          [](const PythonSolver &self) {
            return object_or_none(self.attitude);
          },
          [](PythonSolver &self, py::object value) {
            self.attitude = optional_object(value);
          })
      .def_property(
          "coupled",
          [](const PythonSolver &self) { return object_or_none(self.coupled); },
          [](PythonSolver &self, py::object value) {
            self.coupled = optional_object(value);
          });
}

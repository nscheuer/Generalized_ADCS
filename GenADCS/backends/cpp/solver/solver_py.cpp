#include "coupled_solver.hpp"
#include "solver.hpp"
#include <pybind11/pybind11.h>
#include <optional>
namespace py=pybind11;

namespace {
using PythonSolver = omni::solver::BasicSolver<py::object, py::object,
                                                py::object, py::object>;

std::optional<py::object> optional_object(const py::object& value) {
    if (value.is_none()) return std::nullopt;
    return value;
}

py::object object_or_none(const std::optional<py::object>& value) {
    return value.has_value() ? *value : py::none();
}
}

void bind_solver(py::module_& m) {
    py::class_<PythonSolver>(m, "Solver")
        .def(py::init([](py::object dynamics, py::object orbital,
                         py::object attitude, py::object coupled) {
            if (dynamics.is_none()) {
                dynamics = py::module_::import("GenADCS.backends.cpp.config")
                               .attr("DynamicsConfig")();
            }
            return PythonSolver(std::move(dynamics), optional_object(orbital),
                                optional_object(attitude), optional_object(coupled));
        }), py::arg("dynamics") = py::none(), py::arg("orbital") = py::none(),
            py::arg("attitude") = py::none(), py::arg("coupled") = py::none())
        .def_readwrite("dynamics", &PythonSolver::dynamics)
        .def_property("orbital", [](const PythonSolver& self) {
            return object_or_none(self.orbital);
        }, [](PythonSolver& self, py::object value) {
            self.orbital = optional_object(value);
        })
        .def_property("attitude", [](const PythonSolver& self) {
            return object_or_none(self.attitude);
        }, [](PythonSolver& self, py::object value) {
            self.attitude = optional_object(value);
        })
        .def_property("coupled", [](const PythonSolver& self) {
            return object_or_none(self.coupled);
        }, [](PythonSolver& self, py::object value) {
            self.coupled = optional_object(value);
        });
}

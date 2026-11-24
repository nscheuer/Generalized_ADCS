#include "coupled_solver.hpp"
#include "solver.hpp"
#include <pybind11/pybind11.h>
namespace py=pybind11;

void bind_solver(py::module_& m) {
 py::class_<omni::solver::Solver>(m,"Solver").def(py::init([](py::object d,py::object o,py::object a,py::object c){if(d.is_none())d=py::module_::import("GenADCS.backends.cpp.config").attr("DynamicsConfig")();return omni::solver::Solver(d,o,a,c);}),py::arg("dynamics")=py::none(),py::arg("orbital")=py::none(),py::arg("attitude")=py::none(),py::arg("coupled")=py::none()).def_readwrite("dynamics",&omni::solver::Solver::dynamics).def_readwrite("orbital",&omni::solver::Solver::orbital).def_readwrite("attitude",&omni::solver::Solver::attitude).def_readwrite("coupled",&omni::solver::Solver::coupled);

}

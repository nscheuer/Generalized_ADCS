#include "coupled_solver.hpp"
#include "solver.hpp"
#include <pybind11/pybind11.h>
namespace py=pybind11;
PYBIND11_MODULE(_solver,m){
 py::class_<omni::solver::CoupledSolver>(m,"CoupledSolver").def(py::init<py::object,double,bool,int>(),py::arg("propagator"),py::arg("tf"),py::arg("progress")=false,py::arg("progress_chunk_steps")=500).def_readwrite("propagator",&omni::solver::CoupledSolver::propagator).def_readwrite("tf",&omni::solver::CoupledSolver::tf).def_readwrite("progress",&omni::solver::CoupledSolver::progress).def_readwrite("progress_chunk_steps",&omni::solver::CoupledSolver::progress_chunk_steps).def_property_readonly("dt",&omni::solver::CoupledSolver::dt).def("step",&omni::solver::CoupledSolver::step);
 py::class_<omni::solver::Solver>(m,"Solver").def(py::init([](py::object d,py::object o,py::object a,py::object c){if(d.is_none())d=py::module_::import("GenADCS.backends.cpp.config").attr("DynamicsConfig")();return omni::solver::Solver(d,o,a,c);}),py::arg("dynamics")=py::none(),py::arg("orbital")=py::none(),py::arg("attitude")=py::none(),py::arg("coupled")=py::none()).def_readwrite("dynamics",&omni::solver::Solver::dynamics).def_readwrite("orbital",&omni::solver::Solver::orbital).def_readwrite("attitude",&omni::solver::Solver::attitude).def_readwrite("coupled",&omni::solver::Solver::coupled);
}

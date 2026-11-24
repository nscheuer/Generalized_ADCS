#include "coupled_solver.hpp"
#include "solver.hpp"
#include <pybind11/pybind11.h>
namespace py=pybind11;

void bind_coupled_solver(py::module_& m) {
 py::class_<omni::solver::CoupledSolver>(m,"CoupledSolver").def(py::init<py::object,double,bool,int>(),py::arg("propagator"),py::arg("tf"),py::arg("progress")=false,py::arg("progress_chunk_steps")=500).def_readwrite("propagator",&omni::solver::CoupledSolver::propagator).def_readwrite("tf",&omni::solver::CoupledSolver::tf).def_readwrite("progress",&omni::solver::CoupledSolver::progress).def_readwrite("progress_chunk_steps",&omni::solver::CoupledSolver::progress_chunk_steps).def_property_readonly("dt",&omni::solver::CoupledSolver::dt).def("step",&omni::solver::CoupledSolver::step);

}

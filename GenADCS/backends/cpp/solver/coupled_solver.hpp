#pragma once
#include <pybind11/pybind11.h>
namespace omni::solver {
struct CoupledSolver {pybind11::object propagator;double tf;bool progress;int progress_chunk_steps;CoupledSolver(pybind11::object,double,bool,int);double dt()const;pybind11::object step(pybind11::object,pybind11::object,pybind11::object,pybind11::object)const;};
}

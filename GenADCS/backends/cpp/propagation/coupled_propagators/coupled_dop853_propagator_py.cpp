#include "coupled_rk4_propagator.hpp"
#include "coupled_dop853_propagator.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
namespace omni::propagation {py::tuple integrate_rk4(Vector,double,double,double,const py::object&,bool,std::size_t);py::tuple integrate_dop853(Vector,double,double,double,const py::object&,double,Vector,double,double,bool,std::size_t);}

void bind_coupled_dop853_propagator(py::module_& m) {
m.def("dop853_attempt",&omni::propagation::dop853_attempt);

}

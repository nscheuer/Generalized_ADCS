#include "coupled_rk4_propagator.hpp"
#include "coupled_dop853_propagator.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
namespace omni::propagation {py::tuple integrate_rk4(Vector,double,double,double,const py::object&,bool,std::size_t);py::tuple integrate_dop853(Vector,double,double,double,const py::object&,double,Vector,double,double,bool,std::size_t);}
PYBIND11_MODULE(_coupled,m){m.def("integrate_rk4",&omni::propagation::integrate_rk4,py::arg("y"),py::arg("jd"),py::arg("dt"),py::arg("tf"),py::arg("runtime"),py::arg("progress")=false,py::arg("record_every")=1);m.def("integrate_dop853",&omni::propagation::integrate_dop853,py::arg("y"),py::arg("jd"),py::arg("dt"),py::arg("tf"),py::arg("runtime"),py::arg("rtol"),py::arg("atol"),py::arg("max_step"),py::arg("min_step"),py::arg("progress")=false,py::arg("record_every")=0);m.def("rk4_step",&omni::propagation::rk4_step);m.def("dop853_attempt",&omni::propagation::dop853_attempt);m.def("derivative",&omni::propagation::derivative);m.def("project",[](omni::propagation::Vector y,py::object p){omni::propagation::project(y,p);return y;});}

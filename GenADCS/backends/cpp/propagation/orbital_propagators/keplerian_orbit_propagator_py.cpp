#include "keplerian_orbit_propagator.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using omni::propagation::KeplerianOrbitPropagator;
void bind_keplerian_orbit_propagator(py::module_ &m){
 py::class_<KeplerianOrbitPropagator>(m,"KeplerianOrbitPropagator")
 .def(py::init<double>(),py::arg("dt")).def_property_readonly("dt",&KeplerianOrbitPropagator::dt)
 .def("step",&KeplerianOrbitPropagator::step,py::arg("state"),py::arg("mu"))
 .def("integrate",[](const KeplerianOrbitPropagator &self,const Eigen::VectorXd &state,double mu,double tf,int record_every){return self.integrate(state,mu,tf,record_every);},py::arg("state"),py::arg("mu"),py::arg("tf"),py::arg("record_every")=1);
}

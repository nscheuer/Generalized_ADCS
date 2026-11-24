#include "static.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::V;
using omni::dynamics::Q;
using omni::dynamics::M;

void bind_residual_dipole_model_static(py::module_& m) {
 m.def("static_residual_dipole_torque",[](const V& dipole,const V& field,const Q& q){return omni::dynamics::residual_dipole(dipole,field,q);});
}

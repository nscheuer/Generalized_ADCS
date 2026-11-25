#include "static.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::M;
using omni::dynamics::Q;
using omni::dynamics::V;

void bind_gravity_gradient_model_static(py::module_ &m) {
  m.def("static_gravity_gradient_torque",
        [](const V &p, const Q &q, const M &inertia, double mu) {
          return omni::dynamics::gravity_gradient(p, q, inertia, mu);
        });
}

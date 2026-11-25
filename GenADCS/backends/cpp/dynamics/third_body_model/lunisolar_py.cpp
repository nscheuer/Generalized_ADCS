#include "lunisolar.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::M;
using omni::dynamics::Q;
using omni::dynamics::V;

void bind_third_body_model_lunisolar(py::module_ &m) {
  m.def("lunisolar_acceleration", [](const V &p, const V &sun, const V &moon,
                                     double sun_mu, double moon_mu) {
    return omni::dynamics::lunisolar(p, sun, moon, sun_mu, moon_mu);
  });
}

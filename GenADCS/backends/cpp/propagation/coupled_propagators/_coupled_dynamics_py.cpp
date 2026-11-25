#include "coupled_dop853_propagator.hpp"
#include "coupled_rk4_propagator.hpp"
#include "runtime_python.hpp"
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace omni::propagation;
void bind__coupled_dynamics(py::module_ &m) {
  m.def("derivative", [](Vector y, double jd, py::object p) {
    auto r = runtime_from_python(p);
    return derivative(y, jd, r);
  });
  m.def("project", [](Vector y, py::object p) {
    auto r = runtime_from_python(p);
    project(y, r);
    return y;
  });
}

#include "runtime_python.hpp"
#include "coupled_rk4_propagator.hpp"
#include "coupled_dop853_propagator.hpp"
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace omni::propagation;
void bind_coupled_rk4_propagator(py::module_& m) {
 m.def("rk4_step", [](Vector y,double jd,double dt,py::object p) { auto r=runtime_from_python(p); return rk4_step(y,jd,dt,r); });
}

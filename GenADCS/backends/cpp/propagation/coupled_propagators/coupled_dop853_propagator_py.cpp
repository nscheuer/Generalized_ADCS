#include "runtime_python.hpp"
#include "coupled_rk4_propagator.hpp"
#include "coupled_dop853_propagator.hpp"
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace omni::propagation;
void bind_coupled_dop853_propagator(py::module_& m) {
 m.def("dop853_attempt", [](Vector y,Vector f,double jd,double h,py::object p,double rtol,Vector atol) { auto r=runtime_from_python(p); return dop853_attempt(y,f,jd,h,r,rtol,atol); });
}

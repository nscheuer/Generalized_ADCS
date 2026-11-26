#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_keplerian_orbit_propagator(py::module_ &);
void bind_rk4_attitude_propagator(py::module_ &);
void bind_reference_attitude_propagator(py::module_ &);
void bind_separate_solver(py::module_ &);

PYBIND11_MODULE(_separate, module) {
  bind_keplerian_orbit_propagator(module);
  bind_rk4_attitude_propagator(module);
  bind_reference_attitude_propagator(module);
  bind_separate_solver(module);
}

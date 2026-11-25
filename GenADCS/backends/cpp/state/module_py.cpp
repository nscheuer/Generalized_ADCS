#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_attitude_state(py::module_ &);
void bind_internal_state(py::module_ &);
void bind_orbital_state(py::module_ &);
void bind_state(py::module_ &);
PYBIND11_MODULE(_state, m) {
  bind_attitude_state(m);
  bind_internal_state(m);
  bind_orbital_state(m);
  bind_state(m);
}

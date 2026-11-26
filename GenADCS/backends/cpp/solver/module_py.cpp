#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_coupled_solver(py::module_ &);
void bind_solver(py::module_ &);
void bind_orbital_solver(py::module_ &);
void bind_attitude_solver(py::module_ &);
PYBIND11_MODULE(_solver, m) {
  bind_coupled_solver(m);
  bind_orbital_solver(m);
  bind_attitude_solver(m);
  bind_solver(m);
}

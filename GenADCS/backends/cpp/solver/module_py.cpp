#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_coupled_solver(py::module_ &);
void bind_solver(py::module_ &);
PYBIND11_MODULE(_solver, m) {
  bind_coupled_solver(m);
  bind_solver(m);
}

#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_integrate(py::module_ &);
void bind_coupled_rk4_propagator(py::module_ &);
void bind_coupled_dop853_propagator(py::module_ &);
void bind__coupled_dynamics(py::module_ &);
PYBIND11_MODULE(_coupled, m)
{
    bind_integrate(m);
    bind_coupled_rk4_propagator(m);
    bind_coupled_dop853_propagator(m);
    bind__coupled_dynamics(m);
}

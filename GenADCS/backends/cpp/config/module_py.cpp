#include <pybind11/pybind11.h>
namespace py=pybind11;
void bind_dynamics_config(py::module_&); void bind_simulation_config(py::module_&);
PYBIND11_MODULE(_config,m){bind_dynamics_config(m);bind_simulation_config(m);}

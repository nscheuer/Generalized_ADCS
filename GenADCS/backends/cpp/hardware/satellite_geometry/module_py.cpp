#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_satellite_panel(py::module_ &module);
void bind_satellite_geometry(py::module_ &module);

PYBIND11_MODULE(_satellite_geometry, module) {
  bind_satellite_panel(module);
  bind_satellite_geometry(module);
}

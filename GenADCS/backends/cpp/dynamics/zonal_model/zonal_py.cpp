#include "zonal.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::V;
using omni::dynamics::Q;
using omni::dynamics::M;

void bind_zonal_model_zonal(py::module_& m) {
 m.def("zonal_gravity_force",&omni::dynamics::zonal);
}

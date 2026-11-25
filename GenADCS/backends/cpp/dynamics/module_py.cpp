#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_drag_model_projected_area(py::module_ &);
void bind_srp_model_projected_area(py::module_ &);
void bind_residual_dipole_model_static(py::module_ &);
void bind_gravity_gradient_model_static(py::module_ &);
void bind_third_body_model_lunisolar(py::module_ &);
void bind_zonal_model_zonal(py::module_ &);

PYBIND11_MODULE(_dynamics, module) {
  bind_drag_model_projected_area(module);
  bind_srp_model_projected_area(module);
  bind_residual_dipole_model_static(module);
  bind_gravity_gradient_model_static(module);
  bind_third_body_model_lunisolar(module);
  bind_zonal_model_zonal(module);
}

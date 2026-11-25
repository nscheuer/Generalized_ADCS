#include <pybind11/pybind11.h>
void bind_estimator_base(pybind11::module_ &);
void bind_estimator_mekf(pybind11::module_ &);
PYBIND11_MODULE(_attitude_estimator, m) {
  bind_estimator_base(m);
  bind_estimator_mekf(m);
}

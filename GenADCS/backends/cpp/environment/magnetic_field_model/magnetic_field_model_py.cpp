#include "atmosphere/atmospheric_model.hpp"
#include "environment.hpp"
#include "gravity/j_gravity_model.hpp"
#include "native_models.hpp"
#include "planetary_body/planetary_body.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <array>
#include <string>
#include <utility>

namespace py = pybind11;
namespace env = omni::environment;

void bind_magnetic_field_model_magnetic_field_model(py::module_ &module) {
  py::class_<env::MagneticFieldModel, std::shared_ptr<env::MagneticFieldModel>>(
      module, "MagneticFieldModel")
      .def("field", &env::MagneticFieldModel::field)
      .def("field_inertial_tesla",
           &env::MagneticFieldModel::field_inertial_tesla);
}

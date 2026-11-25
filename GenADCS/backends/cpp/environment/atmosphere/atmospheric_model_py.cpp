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

void bind_atmosphere_atmospheric_model(py::module_ &module) {
  py::class_<env::AtmosphericModel, std::shared_ptr<env::AtmosphericModel>>(
      module, "AtmosphericModel")
      .def("density", &env::AtmosphericModel::density)
      .def_static("velocity", &env::co_rotating_atmosphere_velocity,
                  py::arg("position_inertial_m"), py::arg("epoch_jd"),
                  py::arg("angular_velocity_inertial_rad_s") =
                      Eigen::Vector3d(0.0, 0.0, 7.2921150e-5));
  module.def("atmospheric_density", &env::atmospheric_density);
  module.def("co_rotating_atmosphere_velocity",
             &env::co_rotating_atmosphere_velocity,
             py::arg("position_inertial_m"), py::arg("epoch_jd"),
             py::arg("angular_velocity_inertial_rad_s") =
                 Eigen::Vector3d(0.0, 0.0, 7.2921150e-5));
}

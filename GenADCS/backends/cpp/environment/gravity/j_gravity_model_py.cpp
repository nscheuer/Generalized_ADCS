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

void bind_gravity_j_gravity_model(py::module_ &module) {
  py::class_<env::JGravityModel, std::shared_ptr<env::JGravityModel>>(
      module, "JGravityModel")
      .def(py::init<std::vector<int>, const Eigen::VectorXd &>(),
           py::arg("degrees") = std::vector<int>{2, 3, 4, 5, 6},
           py::arg("coefficients") = Eigen::VectorXd())
      .def("evaluate", &env::JGravityModel::evaluate,
           py::arg("position_inertial_m"), py::arg("epoch_jd") = 0.0)
      .def("acceleration", &env::JGravityModel::acceleration,
           py::arg("position_inertial_m"), py::arg("epoch_jd") = 0.0)
      .def_property_readonly("degrees",
                             [](const env::JGravityModel &self) {
                               py::tuple t(self.degrees().size());
                               for (std::size_t i = 0;
                                    i < self.degrees().size(); ++i)
                                 t[i] = self.degrees()[i];
                               return t;
                             })
      .def_property_readonly("zonal_coefficients",
                             &env::JGravityModel::zonal_coefficients)
      .def_property_readonly("degree", &env::JGravityModel::degree)
      .def_static("acceleration_inertial_m_s2",
                  &env::gravity_acceleration_inertial_m_s2);
  module.attr("JGravityModel").attr("degree") = py::none();
  module.attr("EARTH_ZONAL_COEFFICIENTS") = py::module_::import("numpy").attr(
      "array")(py::make_tuple(1.08262668e-3, -2.5324105e-6, -1.6198976e-6,
                              -2.2729608e-7, 5.406812e-7));
  module.attr("EARTH_MU_M3_S2") = 3.986004418e14;
  module.attr("EARTH_REFERENCE_RADIUS_M") = 6378137.0;
  module.def("gravity_acceleration_inertial_m_s2",
             &env::gravity_acceleration_inertial_m_s2);
}

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

void bind_magnetic_field_model_wmm_wmm(py::module_ &module) {
  py::class_<env::WMM, env::MagneticFieldModel, std::shared_ptr<env::WMM>>(
      module, "WMM")
      .def(py::init<>())
      .def("field",
           [](const env::WMM &self, const Eigen::Vector3d &p, double jd) {
             try {
               return self.field(p, jd);
             } catch (const std::runtime_error &e) {
               PyErr_SetString(PyExc_NotImplementedError, e.what());
               throw py::error_already_set();
             }
           })
      .def("field_inertial_tesla",
           [](const env::WMM &self, const Eigen::Vector3d &p, double jd) {
             try {
               return self.field_inertial_tesla(p, jd);
             } catch (const std::runtime_error &e) {
               PyErr_SetString(PyExc_NotImplementedError, e.what());
               throw py::error_already_set();
             }
           });
}

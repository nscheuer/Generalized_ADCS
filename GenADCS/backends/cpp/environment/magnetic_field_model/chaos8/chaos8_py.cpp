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

void bind_magnetic_field_model_chaos8_chaos8(py::module_ &module) {
  py::class_<env::CHAOS8, env::MagneticFieldModel,
             std::shared_ptr<env::CHAOS8>>(module, "CHAOS8")
      .def(py::init<bool>(), py::arg("include_static") = true)
      .def_property("include_static", &env::CHAOS8::include_static,
                    [](env::CHAOS8 &, bool) {
                      throw py::attribute_error(
                          "include_static is read-only after construction");
                    })
      .def("field", &env::CHAOS8::field)
      .def("field_inertial_tesla", &env::CHAOS8::field_inertial_tesla);
}

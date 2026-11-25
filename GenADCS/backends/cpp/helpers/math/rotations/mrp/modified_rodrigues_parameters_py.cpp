#include "derivatives.hpp"
#include "dynamics.hpp"
#include "rotations/_parameter_base.hpp"
#include "rotations/dcm/dcm.hpp"
#include "rotations/dcm/kernels.hpp"
#include "rotations/mrp/kernels.hpp"
#include "rotations/mrp/modified_rodrigues_parameters.hpp"
#include "rotations/quaternion/algebra.hpp"
#include "rotations/quaternion/conversions.hpp"
#include "rotations/quaternion/interpolation.hpp"
#include "rotations/quaternion/quaternion.hpp"
#include "rotations/rodrigues/kernels.hpp"
#include "rotations/rodrigues/rodrigues_parameters.hpp"
#include "vector.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
namespace math = omni::math;
namespace rotations = omni::rotations;

namespace {
py::object numpy_array(py::object value, py::object dtype) {
  py::object numpy = py::module_::import("numpy");
  if (dtype.is_none()) {
    return numpy.attr("asarray")(value).attr("copy")();
  }
  return numpy.attr("asarray")(value, py::arg("dtype") = dtype).attr("copy")();
}
rotations::ModifiedRodriguesParameters
mrp_from_quaternion_product(const rotations::ModifiedRodriguesParameters &self,
                            const rotations::RotationParameter &other) {
  return rotations::ModifiedRodriguesParameters::from_quaternion(
      self.to_quaternion().multiply(other.to_quaternion()));
}
rotations::ModifiedRodriguesParameters
mrp_relative_to(const rotations::ModifiedRodriguesParameters &self,
                const rotations::RotationParameter &other) {
  return rotations::ModifiedRodriguesParameters::from_quaternion(
      self.to_quaternion().relative_to(other.to_quaternion()));
}
} // namespace
void bind_rotations_mrp_modified_rodrigues_parameters(py::module_ &module) {
  py::class_<rotations::ModifiedRodriguesParameters,
             rotations::RotationParameter>(module,
                                           "Modified_Rodriguez_Parameters")
      .def(py::init<const rotations::Vector3 &>(),
           py::arg("values") = rotations::Vector3::Zero())
      .def_static("identity", &rotations::ModifiedRodriguesParameters::identity)
      .def_static("from_quaternion",
                  &rotations::ModifiedRodriguesParameters::from_quaternion)
      .def_static(
          "from_quaternion_array",
          &rotations::ModifiedRodriguesParameters::from_quaternion_array)
      .def_static("from_axis_angle",
                  &rotations::ModifiedRodriguesParameters::from_axis_angle)
      .def_static("from_rotation_vector",
                  &rotations::ModifiedRodriguesParameters::from_rotation_vector)
      .def_static("from_dcm", &rotations::ModifiedRodriguesParameters::from_dcm)
      .def("switch_shadow_set",
           &rotations::ModifiedRodriguesParameters::switch_shadow_set)
      .def("normalized", &rotations::ModifiedRodriguesParameters::normalized)
      .def("conjugate",
           [](const rotations::ModifiedRodriguesParameters &self) {
             return rotations::ModifiedRodriguesParameters::from_quaternion(
                 self.to_quaternion().conjugate());
           })
      .def("inverse",
           [](const rotations::ModifiedRodriguesParameters &self) {
             return rotations::ModifiedRodriguesParameters::from_quaternion(
                 self.to_quaternion().inverse());
           })
      .def("multiply", &mrp_from_quaternion_product)
      .def("relative_to", &mrp_relative_to)
      .def("__mul__", &mrp_from_quaternion_product)
      .def(
          "__array__",
          [](const rotations::ModifiedRodriguesParameters &self,
             py::object dtype) {
            return numpy_array(py::cast(self.components()), dtype);
          },
          py::arg("dtype") = py::none());

  module.attr("Modified_Rodrigues_Parameters") =
      module.attr("Modified_Rodriguez_Parameters");
  module.attr("ModifiedRodriguesParameters") =
      module.attr("Modified_Rodriguez_Parameters");
}

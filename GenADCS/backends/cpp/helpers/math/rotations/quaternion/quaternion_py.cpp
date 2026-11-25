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
  if (dtype.is_none())
    return numpy.attr("asarray")(value).attr("copy")();
  return numpy.attr("asarray")(value, py::arg("dtype") = dtype).attr("copy")();
}
} // namespace
void bind_rotations_quaternion_quaternion(py::module_ &module) {
  py::class_<rotations::Quaternion>(module, "Quaternion")
      .def(py::init<const rotations::QuaternionVector &, const std::string &>(),
           py::arg("values") = rotations::QuaternionVector(1.0, 0.0, 0.0, 0.0),
           py::arg("convention") = "Hamilton")
      .def_static("identity", &rotations::Quaternion::identity)
      .def_static("from_axis_angle", &rotations::Quaternion::from_axis_angle)
      .def_static("from_rotation_vector",
                  &rotations::Quaternion::from_rotation_vector)
      .def_static("from_dcm", &rotations::Quaternion::from_dcm)
      .def_property_readonly("components", &rotations::Quaternion::components)
      .def_property_readonly("hamilton_components",
                             &rotations::Quaternion::hamilton_components)
      .def_property_readonly("convention", &rotations::Quaternion::convention)
      .def_property_readonly("norm", &rotations::Quaternion::norm)
      .def("as_convention", &rotations::Quaternion::as_convention,
           py::arg("convention") = "Hamilton")
      .def("to_convention", &rotations::Quaternion::to_convention,
           py::arg("convention") = "Hamilton")
      .def("normalized", &rotations::Quaternion::normalized)
      .def("conjugate", &rotations::Quaternion::conjugate)
      .def("inverse", &rotations::Quaternion::inverse)
      .def("multiply", &rotations::Quaternion::multiply)
      .def("rotate", &rotations::Quaternion::rotate)
      .def("inertial_to_body", &rotations::Quaternion::inertial_to_body)
      .def("body_to_inertial", &rotations::Quaternion::body_to_inertial)
      .def("to_dcm", &rotations::Quaternion::to_dcm)
      .def("relative_to", &rotations::Quaternion::relative_to)
      .def("__mul__", &rotations::Quaternion::multiply)
      .def(
          "__array__",
          [](const rotations::Quaternion &self, py::object dtype) {
            return numpy_array(py::cast(self.hamilton_components()), dtype);
          },
          py::arg("dtype") = py::none())
      .def("__repr__", [](const rotations::Quaternion &self) {
        return "Quaternion(" +
               py::str(py::cast(self.hamilton_components()))
                   .cast<std::string>() +
               ", convention='Hamilton', frame='q_B<-I')";
      });
}

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

void bind_rotations__parameter_base(py::module_ &module) {
  py::class_<rotations::RotationParameter>(module, "_RotationParameter")
      .def_property_readonly("components",
                             &rotations::RotationParameter::components)
      .def_property_readonly("hamilton_components",
                             &rotations::RotationParameter::hamilton_components)
      .def_property_readonly("norm", &rotations::RotationParameter::norm)
      .def("as_vector", &rotations::RotationParameter::as_vector)
      .def("to_quaternion", &rotations::RotationParameter::to_quaternion)
      .def("to_dcm", &rotations::RotationParameter::to_dcm)
      .def("rotate", &rotations::RotationParameter::rotate)
      .def("inertial_to_body", &rotations::RotationParameter::inertial_to_body)
      .def("body_to_inertial", &rotations::RotationParameter::body_to_inertial);
}

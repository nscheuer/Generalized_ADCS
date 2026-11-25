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

void bind_rotations_quaternion_conversions(py::module_ &module) {
  module.def("_quaternion_to_dcm", &rotations::quaternion_to_dcm);
  module.def("_dcm_to_quaternion", &rotations::dcm_to_quaternion);
  module.def("_axis_angle_to_quaternion", &rotations::axis_angle_to_quaternion);
  module.def("rotation_vector_to_quaternion",
             &rotations::rotation_vector_to_quaternion);
  module.def("quaternion_to_rotation_vector",
             &rotations::quaternion_to_rotation_vector);
}

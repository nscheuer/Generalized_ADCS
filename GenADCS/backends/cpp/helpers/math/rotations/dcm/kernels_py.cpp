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

void bind_rotations_dcm_kernels(py::module_ &module) {
  module.def("axis_angle_to_dcm", &rotations::axis_angle_to_dcm);
  module.def(
      "rotate_vector",
      py::overload_cast<const rotations::Matrix3 &, const rotations::Vector3 &>(
          &rotations::rotate_vector));
  module.def("inverse_dcm", &rotations::inverse_dcm);
  module.def("compose_dcm", &rotations::compose_dcm);
  module.def("relative_dcm", &rotations::relative_dcm);
  module.def("dcm_orthogonality_error", &rotations::dcm_orthogonality_error);
  module.def("normalize_dcm", &rotations::normalize_dcm);
}

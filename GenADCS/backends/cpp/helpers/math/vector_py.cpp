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

void bind_vector(py::module_ &module) {
  module.def("cross", &math::cross);
  module.def("norm", &math::norm);
  module.def("normalize", &math::normalize);
  module.def("unit", &math::unit);
  module.def("skew", &math::skew);
}

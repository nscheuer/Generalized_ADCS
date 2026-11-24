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

void bind_rotations_quaternion_algebra(py::module_ &module)
{
    module.def("_normalize_quaternion", &rotations::normalize_quaternion);
    module.def("_quaternion_norm", &rotations::quaternion_norm);
    module.def("_jpl_to_hamilton", &rotations::jpl_to_hamilton);
    module.def("_hamilton_to_jpl", &rotations::hamilton_to_jpl);
    module.def("_conjugate", &rotations::conjugate);
    module.def("_inverse", &rotations::inverse);
    module.def("_multiply", &rotations::multiply);
    module.def("_relative_quaternion", &rotations::relative_quaternion);
    module.def("_rotate_vector", py::overload_cast<const rotations::QuaternionVector &, const rotations::Vector3 &>(
                                     &rotations::rotate_vector));
}

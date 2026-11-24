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

void bind_derivatives(py::module_ &module)
{
    module.def("quaternion_derivative", &math::quaternion_derivative);
    module.def("quaternion_rate_matrix", &math::quaternion_rate_matrix);
    module.def("angular_velocity_from_quaternion_derivative", &math::angular_velocity_from_quaternion_derivative);
    module.def("quaternion_normalization_jacobian", &math::quaternion_normalization_jacobian);
    module.def("rotated_vector_quaternion_jacobian", &math::rotated_vector_quaternion_jacobian);
    module.def("quaternion_to_euler321", &math::quaternion_to_euler321);
}

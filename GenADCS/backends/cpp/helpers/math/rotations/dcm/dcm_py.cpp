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

namespace
{
py::object numpy_array(py::object value, py::object dtype)
{
    py::object numpy = py::module_::import("numpy");
    if (dtype.is_none())
        return numpy.attr("asarray")(value).attr("copy")();
    return numpy.attr("asarray")(value, py::arg("dtype") = dtype).attr("copy")();
}
} // namespace
void bind_rotations_dcm_dcm(py::module_ &module)
{
    py::class_<rotations::DCM>(module, "DCM")
        .def(py::init<const rotations::Matrix3 &>(), py::arg("matrix") = rotations::Matrix3::Identity())
        .def_static("identity", &rotations::DCM::identity)
        .def_static("from_axis_angle", &rotations::DCM::from_axis_angle)
        .def_static("from_rotation_vector", &rotations::DCM::from_rotation_vector)
        .def_static("from_quaternion", &rotations::DCM::from_quaternion)
        .def_static("from_dcm", &rotations::DCM::from_dcm)
        .def_property_readonly("components", &rotations::DCM::components)
        .def_property_readonly("hamilton_components", &rotations::DCM::hamilton_components)
        .def_property_readonly("norm", &rotations::DCM::norm)
        .def_property_readonly("orthogonality_error", &rotations::DCM::orthogonality_error)
        .def("as_matrix", &rotations::DCM::as_matrix)
        .def("to_dcm", &rotations::DCM::to_dcm)
        .def("to_quaternion", &rotations::DCM::to_quaternion)
        .def("normalized", &rotations::DCM::normalized)
        .def("inverse", &rotations::DCM::inverse)
        .def("conjugate", &rotations::DCM::conjugate)
        .def("multiply", &rotations::DCM::multiply)
        .def("rotate", &rotations::DCM::rotate)
        .def("inertial_to_body", &rotations::DCM::inertial_to_body)
        .def("body_to_inertial", &rotations::DCM::body_to_inertial)
        .def("relative_to", &rotations::DCM::relative_to)
        .def("__mul__", &rotations::DCM::multiply)
        .def(
            "__array__",
            [](const rotations::DCM &self, py::object dtype) { return numpy_array(py::cast(self.as_matrix()), dtype); },
            py::arg("dtype") = py::none());
}

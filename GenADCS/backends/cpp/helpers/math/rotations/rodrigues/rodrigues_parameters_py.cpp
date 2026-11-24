#include "derivatives.hpp"
#include "dynamics.hpp"
#include "rotations/_parameter_base.hpp"
#include "rotations/quaternion/algebra.hpp"
#include "rotations/quaternion/conversions.hpp"
#include "rotations/quaternion/interpolation.hpp"
#include "rotations/quaternion/quaternion.hpp"
#include "rotations/dcm/dcm.hpp"
#include "rotations/dcm/kernels.hpp"
#include "rotations/mrp/kernels.hpp"
#include "rotations/mrp/modified_rodrigues_parameters.hpp"
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
    if (dtype.is_none()) return numpy.attr("asarray")(value).attr("copy")();
    return numpy.attr("asarray")(value, py::arg("dtype") = dtype).attr("copy")();
}
rotations::RodriguesParameters rodrigues_from_quaternion_product(
        const rotations::RodriguesParameters& self,
        const rotations::RotationParameter& other) {
    return rotations::RodriguesParameters::from_quaternion(self.to_quaternion().multiply(other.to_quaternion()));
}
rotations::RodriguesParameters rodrigues_relative_to(
        const rotations::RodriguesParameters& self,
        const rotations::RotationParameter& other) {
    return rotations::RodriguesParameters::from_quaternion(self.to_quaternion().relative_to(other.to_quaternion()));
}
}
void bind_rotations_rodrigues_rodrigues_parameters(py::module_& module) {
    py::class_<rotations::RodriguesParameters, rotations::RotationParameter>(module, "Rodriguez_Parameters")
        .def(py::init<const rotations::Vector3&>(), py::arg("values") = rotations::Vector3::Zero())
        .def_static("identity", &rotations::RodriguesParameters::identity)
        .def_static("from_quaternion", &rotations::RodriguesParameters::from_quaternion)
        .def_static("from_quaternion_array", &rotations::RodriguesParameters::from_quaternion_array)
        .def_static("from_axis_angle", &rotations::RodriguesParameters::from_axis_angle)
        .def_static("from_rotation_vector", &rotations::RodriguesParameters::from_rotation_vector)
        .def_static("from_dcm", &rotations::RodriguesParameters::from_dcm)
        .def("normalized", &rotations::RodriguesParameters::normalized)
        .def("conjugate", [](const rotations::RodriguesParameters& self) { return rotations::RodriguesParameters::from_quaternion(self.to_quaternion().conjugate()); })
        .def("inverse", [](const rotations::RodriguesParameters& self) { return rotations::RodriguesParameters::from_quaternion(self.to_quaternion().inverse()); })
        .def("multiply", &rodrigues_from_quaternion_product)
        .def("relative_to", &rodrigues_relative_to)
        .def("__mul__", &rodrigues_from_quaternion_product)
        .def("__array__", [](const rotations::RodriguesParameters& self, py::object dtype) { return numpy_array(py::cast(self.components()), dtype); }, py::arg("dtype") = py::none());

    module.attr("Rodrigues_Parameters") = module.attr("Rodriguez_Parameters");
    module.attr("RodriguesParameters") = module.attr("Rodriguez_Parameters");
}

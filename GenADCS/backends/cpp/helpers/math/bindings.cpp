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
#include "time/conversions.hpp"
#include "vector.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
namespace math = omni::math;
namespace rotations = omni::rotations;
namespace timeconv = omni::time;

namespace {
py::object numpy_array(py::object value, py::object dtype) {
    py::object numpy = py::module_::import("numpy");
    if (dtype.is_none()) return numpy.attr("asarray")(value).attr("copy")();
    return numpy.attr("asarray")(value, py::arg("dtype") = dtype).attr("copy")();
}

rotations::Vector3 vector3_from_array(py::object value) {
    return value.cast<rotations::Vector3>();
}

rotations::QuaternionVector quaternion_from_array(py::object value) {
    return value.cast<rotations::QuaternionVector>();
}

rotations::Matrix3 matrix3_from_array(py::object value) {
    return value.cast<rotations::Matrix3>();
}

rotations::ModifiedRodriguesParameters mrp_from_quaternion_product(
        const rotations::ModifiedRodriguesParameters& self,
        const rotations::RotationParameter& other) {
    return rotations::ModifiedRodriguesParameters::from_quaternion(self.to_quaternion().multiply(other.to_quaternion()));
}

rotations::ModifiedRodriguesParameters mrp_relative_to(
        const rotations::ModifiedRodriguesParameters& self,
        const rotations::RotationParameter& other) {
    return rotations::ModifiedRodriguesParameters::from_quaternion(self.to_quaternion().relative_to(other.to_quaternion()));
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

PYBIND11_MODULE(_helpers, module) {
    module.doc() = "Eigen-backed C++ ADCS Omni mathematical helpers";

    module.def("cross", &math::cross);
    module.def("norm", &math::norm);
    module.def("normalize", &math::normalize);
    module.def("unit", &math::unit);
    module.def("skew", &math::skew);

    module.def("quaternion_derivative", &math::quaternion_derivative);
    module.def("quaternion_rate_matrix", &math::quaternion_rate_matrix);
    module.def("angular_velocity_from_quaternion_derivative", &math::angular_velocity_from_quaternion_derivative);
    module.def("quaternion_normalization_jacobian", &math::quaternion_normalization_jacobian);
    module.def("rotated_vector_quaternion_jacobian", &math::rotated_vector_quaternion_jacobian);
    module.def("quaternion_to_euler321", &math::quaternion_to_euler321);
    module.def("rigid_body_angular_acceleration", &math::rigid_body_angular_acceleration);
    module.def("wheel_coupled_angular_acceleration", &math::wheel_coupled_angular_acceleration);
    module.def("wheel_momentum_derivative", &math::wheel_momentum_derivative);

    module.def("_normalize_quaternion", &rotations::normalize_quaternion);
    module.def("_quaternion_norm", &rotations::quaternion_norm);
    module.def("_jpl_to_hamilton", &rotations::jpl_to_hamilton);
    module.def("_hamilton_to_jpl", &rotations::hamilton_to_jpl);
    module.def("_conjugate", &rotations::conjugate);
    module.def("_inverse", &rotations::inverse);
    module.def("_multiply", &rotations::multiply);
    module.def("_relative_quaternion", &rotations::relative_quaternion);
    module.def("_rotate_vector", py::overload_cast<const rotations::QuaternionVector&, const rotations::Vector3&>(&rotations::rotate_vector));
    module.def("_quaternion_to_dcm", &rotations::quaternion_to_dcm);
    module.def("_dcm_to_quaternion", &rotations::dcm_to_quaternion);
    module.def("_axis_angle_to_quaternion", &rotations::axis_angle_to_quaternion);
    module.def("slerp", &rotations::slerp);
    module.def("rotation_vector_to_quaternion", &rotations::rotation_vector_to_quaternion);
    module.def("quaternion_to_rotation_vector", &rotations::quaternion_to_rotation_vector);

    py::class_<rotations::Quaternion>(module, "Quaternion")
        .def(py::init<const rotations::QuaternionVector&, const std::string&>(),
             py::arg("values") = rotations::QuaternionVector(1.0, 0.0, 0.0, 0.0), py::arg("convention") = "Hamilton")
        .def_static("identity", &rotations::Quaternion::identity)
        .def_static("from_axis_angle", &rotations::Quaternion::from_axis_angle)
        .def_static("from_rotation_vector", &rotations::Quaternion::from_rotation_vector)
        .def_static("from_dcm", &rotations::Quaternion::from_dcm)
        .def_property_readonly("components", &rotations::Quaternion::components)
        .def_property_readonly("hamilton_components", &rotations::Quaternion::hamilton_components)
        .def_property_readonly("convention", &rotations::Quaternion::convention)
        .def_property_readonly("norm", &rotations::Quaternion::norm)
        .def("as_convention", &rotations::Quaternion::as_convention, py::arg("convention") = "Hamilton")
        .def("to_convention", &rotations::Quaternion::to_convention, py::arg("convention") = "Hamilton")
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
        .def("__array__", [](const rotations::Quaternion& self, py::object dtype) {
            return numpy_array(py::cast(self.hamilton_components()), dtype);
        }, py::arg("dtype") = py::none())
        .def("__repr__", [](const rotations::Quaternion& self) {
            return "Quaternion(" + py::str(py::cast(self.hamilton_components())).cast<std::string>()
                + ", convention='Hamilton', frame='q_B<-I')";
        });

    module.def("axis_angle_to_dcm", &rotations::axis_angle_to_dcm);
    module.def("rotate_vector", py::overload_cast<const rotations::Matrix3&, const rotations::Vector3&>(&rotations::rotate_vector));
    module.def("inverse_dcm", &rotations::inverse_dcm);
    module.def("compose_dcm", &rotations::compose_dcm);
    module.def("relative_dcm", &rotations::relative_dcm);
    module.def("dcm_orthogonality_error", &rotations::dcm_orthogonality_error);
    module.def("normalize_dcm", &rotations::normalize_dcm);

    py::class_<rotations::DCM>(module, "DCM")
        .def(py::init<const rotations::Matrix3&>(), py::arg("matrix") = rotations::Matrix3::Identity())
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
        .def("__array__", [](const rotations::DCM& self, py::object dtype) {
            return numpy_array(py::cast(self.as_matrix()), dtype);
        }, py::arg("dtype") = py::none());

    module.def("mrp_to_quaternion", &rotations::mrp_to_quaternion_parameters);
    module.def("quaternion_to_mrp", &rotations::quaternion_to_mrp_parameters);
    module.def("mrp_shadow_set", &rotations::mrp_shadow_set);
    module.def("rodrigues_to_quaternion", &rotations::rodrigues_to_quaternion_parameters);
    module.def("quaternion_to_rodrigues", &rotations::quaternion_to_rodrigues_parameters);

    py::class_<rotations::RotationParameter>(module, "_RotationParameter")
        .def_property_readonly("components", &rotations::RotationParameter::components)
        .def_property_readonly("hamilton_components", &rotations::RotationParameter::hamilton_components)
        .def_property_readonly("norm", &rotations::RotationParameter::norm)
        .def("as_vector", &rotations::RotationParameter::as_vector)
        .def("to_quaternion", &rotations::RotationParameter::to_quaternion)
        .def("to_dcm", &rotations::RotationParameter::to_dcm)
        .def("rotate", &rotations::RotationParameter::rotate)
        .def("inertial_to_body", &rotations::RotationParameter::inertial_to_body)
        .def("body_to_inertial", &rotations::RotationParameter::body_to_inertial);

    py::class_<rotations::ModifiedRodriguesParameters, rotations::RotationParameter>(module, "Modified_Rodriguez_Parameters")
        .def(py::init<const rotations::Vector3&>(), py::arg("values") = rotations::Vector3::Zero())
        .def_static("identity", &rotations::ModifiedRodriguesParameters::identity)
        .def_static("from_quaternion", &rotations::ModifiedRodriguesParameters::from_quaternion)
        .def_static("from_quaternion_array", &rotations::ModifiedRodriguesParameters::from_quaternion_array)
        .def_static("from_axis_angle", &rotations::ModifiedRodriguesParameters::from_axis_angle)
        .def_static("from_rotation_vector", &rotations::ModifiedRodriguesParameters::from_rotation_vector)
        .def_static("from_dcm", &rotations::ModifiedRodriguesParameters::from_dcm)
        .def("switch_shadow_set", &rotations::ModifiedRodriguesParameters::switch_shadow_set)
        .def("normalized", &rotations::ModifiedRodriguesParameters::normalized)
        .def("conjugate", [](const rotations::ModifiedRodriguesParameters& self) { return rotations::ModifiedRodriguesParameters::from_quaternion(self.to_quaternion().conjugate()); })
        .def("inverse", [](const rotations::ModifiedRodriguesParameters& self) { return rotations::ModifiedRodriguesParameters::from_quaternion(self.to_quaternion().inverse()); })
        .def("multiply", &mrp_from_quaternion_product)
        .def("relative_to", &mrp_relative_to)
        .def("__mul__", &mrp_from_quaternion_product)
        .def("__array__", [](const rotations::ModifiedRodriguesParameters& self, py::object dtype) { return numpy_array(py::cast(self.components()), dtype); }, py::arg("dtype") = py::none());

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

    module.def("calendar_to_jd", &timeconv::calendar_to_jd,
        py::arg("year"), py::arg("month"), py::arg("day"), py::arg("hour") = 0,
        py::arg("minute") = 0, py::arg("second") = 0.0);
    module.def("jd_to_calendar", &timeconv::jd_to_calendar);
    module.def("utc_jd_to_tai_jd", &timeconv::utc_jd_to_tai_jd);
    module.def("tai_jd_to_utc_jd", &timeconv::tai_jd_to_utc_jd);
    module.def("tai_jd_to_tt_jd", &timeconv::tai_jd_to_tt_jd);
    module.def("tt_jd_to_tai_jd", &timeconv::tt_jd_to_tai_jd);
    module.def("utc_jd_to_tt_jd", &timeconv::utc_jd_to_tt_jd);
    module.def("tt_jd_to_utc_jd", &timeconv::tt_jd_to_utc_jd);
    module.def("utc_jd_to_ut1_jd", &timeconv::utc_jd_to_ut1_jd);
    module.def("ut1_jd_to_utc_jd", &timeconv::ut1_jd_to_utc_jd);
    module.def("tdb_jd_to_tt_jd", &timeconv::tdb_jd_to_tt_jd);
    module.def("utc_jd_to_tdb_jd", &timeconv::utc_jd_to_tdb_jd);
    module.def("tdb_jd_to_utc_jd", &timeconv::tdb_jd_to_utc_jd);
    module.def("tai_jd_to_ut1_jd", &timeconv::tai_jd_to_ut1_jd);
    module.def("ut1_jd_to_tai_jd", &timeconv::ut1_jd_to_tai_jd);
    module.def("tt_jd_to_ut1_jd", &timeconv::tt_jd_to_ut1_jd);
    module.def("ut1_jd_to_tt_jd", &timeconv::ut1_jd_to_tt_jd);
    module.def("tt_jd_to_tdb_jd", &timeconv::tt_jd_to_tdb_jd);
    module.def("tai_jd_to_tdb_jd", &timeconv::tai_jd_to_tdb_jd);
    module.def("tdb_jd_to_tai_jd", &timeconv::tdb_jd_to_tai_jd);
    module.def("ut1_jd_to_tdb_jd", &timeconv::ut1_jd_to_tdb_jd);
    module.def("tdb_jd_to_ut1_jd", &timeconv::tdb_jd_to_ut1_jd);
    module.def("jd_to_utc_calendar_fields", &timeconv::jd_to_utc_calendar_fields);
    module.def("day_of_year", &timeconv::day_of_year);

    module.attr("Modified_Rodrigues_Parameters") = module.attr("Modified_Rodriguez_Parameters");
    module.attr("ModifiedRodriguesParameters") = module.attr("Modified_Rodriguez_Parameters");
    module.attr("Rodrigues_Parameters") = module.attr("Rodriguez_Parameters");
    module.attr("RodriguesParameters") = module.attr("Rodriguez_Parameters");
}

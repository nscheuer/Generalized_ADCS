#include "conversions.hpp"
#include "frames.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace omni_frames = omni::frames;

namespace {
py::array_t<double> vector_to_array(const omni_frames::Vector3& vector) {
    py::array_t<double> result(3);
    auto output = result.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < 3; ++index) {
        output(index) = vector[index];
    }
    return result;
}

py::array_t<double> matrix_to_array(const omni_frames::Matrix3& matrix) {
    py::array_t<double> result({3, 3});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < 3; ++row) {
        for (py::ssize_t column = 0; column < 3; ++column) {
            output(row, column) = matrix(row, column);
        }
    }
    return result;
}

py::array_t<double> as_array(py::handle value, const char* name, py::ssize_t rank,
                             py::ssize_t first_extent, py::ssize_t second_extent = -1) {
    py::array array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw py::type_error(std::string(name) + " must be convertible to a float64 array");
    }
    if ((array.ndim() != rank) || (array.shape(0) != first_extent) ||
        ((rank == 2) && (array.shape(1) != second_extent))) {
        throw py::value_error(std::string(name) + " has an invalid shape");
    }
    py::object numpy = py::module_::import("numpy");
    if (!numpy.attr("all")(numpy.attr("isfinite")(array)).cast<bool>()) {
        throw py::value_error(std::string(name) + " must contain only finite values");
    }
    return py::reinterpret_borrow<py::array_t<double>>(array);
}

template <int Size>
Eigen::Matrix<double, Size, 1> to_fixed_vector(py::handle value, const char* name) {
    auto array = as_array(value, name, 1, Size);
    auto input = array.unchecked<1>();
    Eigen::Matrix<double, Size, 1> result;
    for (int index = 0; index < Size; ++index) {
        result[index] = input(index);
    }
    return result;
}

omni_frames::Matrix3 to_matrix(py::handle value, const char* name) {
    auto array = as_array(value, name, 2, 3, 3);
    auto input = array.unchecked<2>();
    omni_frames::Matrix3 result;
    for (py::ssize_t row = 0; row < 3; ++row) {
        for (py::ssize_t column = 0; column < 3; ++column) {
            result(row, column) = input(row, column);
        }
    }
    return result;
}

std::optional<double> optional_double(py::object value) {
    if (value.is_none()) return std::nullopt;
    const double result = value.cast<double>();
    if (!std::isfinite(result)) throw py::value_error("frame context values must be finite");
    return result;
}

omni_frames::FrameContext make_context(py::object jd_ut1, py::object jd_tt,
                                       double xp, double yp,
                                       py::object position_gcrf,
                                       py::object velocity_gcrf,
                                       py::object quaternion) {
    omni_frames::FrameContext context;
    context.jd_ut1 = optional_double(jd_ut1);
    context.jd_tt = optional_double(jd_tt);
    context.xp = xp;
    context.yp = yp;
    if (!std::isfinite(xp) || !std::isfinite(yp)) {
        throw py::value_error("xp and yp must be finite");
    }
    if (!position_gcrf.is_none()) context.position_gcrf = to_fixed_vector<3>(position_gcrf, "position_gcrf");
    if (!velocity_gcrf.is_none()) context.velocity_gcrf = to_fixed_vector<3>(velocity_gcrf, "velocity_gcrf");
    if (!quaternion.is_none()) context.quaternion_body_from_gcrf = to_fixed_vector<4>(quaternion, "q_body_from_gcrf");
    return context;
}

py::array frame_rotation(const std::string& source, const std::string& target,
                         py::object jd_ut1, py::object jd_tt, double xp, double yp,
                         py::object position_gcrf, py::object velocity_gcrf,
                         py::object quaternion) {
    const auto context = make_context(jd_ut1, jd_tt, xp, yp,
                                      position_gcrf, velocity_gcrf, quaternion);
    return matrix_to_array(omni_frames::frame_rotation(omni_frames::parse_frame(source),
                                                        omni_frames::parse_frame(target), context));
}

py::array convert_vector(py::object vector, const std::string& source,
                         const std::string& target, py::object jd_ut1,
                         py::object jd_tt, double xp, double yp,
                         py::object position_gcrf, py::object velocity_gcrf,
                         py::object quaternion) {
    const auto context = make_context(jd_ut1, jd_tt, xp, yp,
                                      position_gcrf, velocity_gcrf, quaternion);
    const auto vector_value = to_fixed_vector<3>(vector, "vector");
    return vector_to_array(omni_frames::convert_vector(vector_value,
                                                        omni_frames::parse_frame(source),
                                                        omni_frames::parse_frame(target), context));
}
}

PYBIND11_MODULE(_frames, module) {
    module.doc() = "MISRA-oriented C++ reference-frame transforms";

    module.def("inertial_to_body_matrix", [](py::object quaternion) {
        return matrix_to_array(omni_frames::inertial_to_body_matrix(
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
    }, py::arg("q_b_from_i"));
    module.def("body_to_inertial_matrix", [](py::object quaternion) {
        return matrix_to_array(omni_frames::body_to_inertial_matrix(
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
    }, py::arg("q_b_from_i"));
    module.def("inertial_to_body", [](py::object vector, py::object quaternion) {
        return vector_to_array(omni_frames::inertial_to_body(
            to_fixed_vector<3>(vector, "vector_i"), to_fixed_vector<4>(quaternion, "q_b_from_i")));
    }, py::arg("vector_i"), py::arg("q_b_from_i"));
    module.def("body_to_inertial", [](py::object vector, py::object quaternion) {
        return vector_to_array(omni_frames::body_to_inertial(
            to_fixed_vector<3>(vector, "vector_b"), to_fixed_vector<4>(quaternion, "q_b_from_i")));
    }, py::arg("vector_b"), py::arg("q_b_from_i"));
    module.def("transform_vector", [](py::object vector, py::object matrix) {
        return vector_to_array(omni_frames::transform_vector(
            to_fixed_vector<3>(vector, "vector"), to_matrix(matrix, "source_to_target_dcm")));
    }, py::arg("vector"), py::arg("source_to_target_dcm"));
    module.def("transform_covariance", [](py::object covariance, py::object matrix) {
        return matrix_to_array(omni_frames::transform_covariance(
            to_matrix(covariance, "covariance_source"), to_matrix(matrix, "source_to_target_dcm")));
    }, py::arg("covariance_source"), py::arg("source_to_target_dcm"));

    module.def("gcrf_to_lvlh_matrix", [](py::object position, py::object velocity) {
        return matrix_to_array(omni_frames::gcrf_to_lvlh_matrix(
            to_fixed_vector<3>(position, "position_gcrf"), to_fixed_vector<3>(velocity, "velocity_gcrf")));
    }, py::arg("position_gcrf"), py::arg("velocity_gcrf"));
    module.def("gcrf_to_ecef_matrix", [](double jd_ut1, double jd_tt, double xp, double yp) {
        return matrix_to_array(omni_frames::gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp));
    }, py::arg("jd_ut1"), py::arg("jd_tt"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
    module.def("ecef_to_gcrf_matrix", [](double jd_ut1, double jd_tt, double xp, double yp) {
        return matrix_to_array(omni_frames::ecef_to_gcrf_matrix(jd_ut1, jd_tt, xp, yp));
    }, py::arg("jd_ut1"), py::arg("jd_tt"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
    module.def("gcrf_to_teme_matrix", [](double jd_tt) {
        return matrix_to_array(omni_frames::gcrf_to_teme_matrix(jd_tt));
    }, py::arg("jd_tt"));
    module.def("teme_to_gcrf_matrix", [](double jd_tt) {
        return matrix_to_array(omni_frames::teme_to_gcrf_matrix(jd_tt));
    }, py::arg("jd_tt"));
    module.def("teme_to_ecef_matrix", [](double jd_ut1, double xp, double yp) {
        return matrix_to_array(omni_frames::teme_to_ecef_matrix(jd_ut1, xp, yp));
    }, py::arg("jd_ut1"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
    module.def("ecef_to_teme_matrix", [](double jd_ut1, double xp, double yp) {
        return matrix_to_array(omni_frames::ecef_to_teme_matrix(jd_ut1, xp, yp));
    }, py::arg("jd_ut1"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
    module.def("frame_rotation", &frame_rotation,
        py::arg("source"), py::arg("target"), py::kw_only(),
        py::arg("jd_ut1") = py::none(), py::arg("jd_tt") = py::none(),
        py::arg("xp") = 0.0, py::arg("yp") = 0.0,
        py::arg("position_gcrf") = py::none(), py::arg("velocity_gcrf") = py::none(),
        py::arg("q_body_from_gcrf") = py::none());
    module.def("convert_vector", &convert_vector,
        py::arg("vector"), py::arg("source"), py::arg("target"), py::kw_only(),
        py::arg("jd_ut1") = py::none(), py::arg("jd_tt") = py::none(),
        py::arg("xp") = 0.0, py::arg("yp") = 0.0,
        py::arg("position_gcrf") = py::none(), py::arg("velocity_gcrf") = py::none(),
        py::arg("q_body_from_gcrf") = py::none());
    module.def("ecef_to_geodetic", [](py::object position) {
        return vector_to_array(omni_frames::ecef_to_geodetic(
            to_fixed_vector<3>(position, "position_ecef_m")));
    }, py::arg("position_ecef_m"));
    module.def("gcrf_position_to_geodetic", [](py::object position, double jd_ut1,
                                                double jd_tt, double xp, double yp) {
        return vector_to_array(omni_frames::gcrf_position_to_geodetic(
            to_fixed_vector<3>(position, "position_gcrf_m"), jd_ut1, jd_tt, xp, yp));
    }, py::arg("position_gcrf_m"), py::arg("jd_ut1"), py::arg("jd_tt"),
       py::arg("xp") = 0.0, py::arg("yp") = 0.0);
}

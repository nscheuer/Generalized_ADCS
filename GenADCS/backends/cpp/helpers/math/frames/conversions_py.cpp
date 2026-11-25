#include "bindings_common.hpp"
#include "conversions.hpp"

#include <cmath>
#include <optional>

using namespace frame_bindings;

namespace {
std::optional<double> optional_double(py::object value) {
  if (value.is_none())
    return std::nullopt;
  const double result = value.cast<double>();
  if (!std::isfinite(result))
    throw py::value_error("frame context values must be finite");
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
  if (!position_gcrf.is_none())
    context.position_gcrf = to_fixed_vector<3>(position_gcrf, "position_gcrf");
  if (!velocity_gcrf.is_none())
    context.velocity_gcrf = to_fixed_vector<3>(velocity_gcrf, "velocity_gcrf");
  if (!quaternion.is_none())
    context.quaternion_body_from_gcrf =
        to_fixed_vector<4>(quaternion, "q_body_from_gcrf");
  return context;
}

py::array frame_rotation(const std::string &source, const std::string &target,
                         py::object jd_ut1, py::object jd_tt, double xp,
                         double yp, py::object position_gcrf,
                         py::object velocity_gcrf, py::object quaternion) {
  const auto context = make_context(jd_ut1, jd_tt, xp, yp, position_gcrf,
                                    velocity_gcrf, quaternion);
  return matrix_to_array(
      omni_frames::frame_rotation(omni_frames::parse_frame(source),
                                  omni_frames::parse_frame(target), context));
}

py::array convert_vector(py::object vector, const std::string &source,
                         const std::string &target, py::object jd_ut1,
                         py::object jd_tt, double xp, double yp,
                         py::object position_gcrf, py::object velocity_gcrf,
                         py::object quaternion) {
  const auto context = make_context(jd_ut1, jd_tt, xp, yp, position_gcrf,
                                    velocity_gcrf, quaternion);
  const auto vector_value = to_fixed_vector<3>(vector, "vector");
  return vector_to_array(omni_frames::convert_vector(
      vector_value, omni_frames::parse_frame(source),
      omni_frames::parse_frame(target), context));
}
} // namespace

void bind_conversions(py::module_ &module) {
  module.def(
      "gcrf_to_lvlh_matrix",
      [](py::object position, py::object velocity) {
        return matrix_to_array(omni_frames::gcrf_to_lvlh_matrix(
            to_fixed_vector<3>(position, "position_gcrf"),
            to_fixed_vector<3>(velocity, "velocity_gcrf")));
      },
      py::arg("position_gcrf"), py::arg("velocity_gcrf"));
  module.def(
      "gcrf_to_ecef_matrix",
      [](double jd_ut1, double jd_tt, double xp, double yp) {
        return matrix_to_array(
            omni_frames::gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp));
      },
      py::arg("jd_ut1"), py::arg("jd_tt"), py::arg("xp") = 0.0,
      py::arg("yp") = 0.0);
  module.def(
      "ecef_to_gcrf_matrix",
      [](double jd_ut1, double jd_tt, double xp, double yp) {
        return matrix_to_array(
            omni_frames::ecef_to_gcrf_matrix(jd_ut1, jd_tt, xp, yp));
      },
      py::arg("jd_ut1"), py::arg("jd_tt"), py::arg("xp") = 0.0,
      py::arg("yp") = 0.0);
  module.def(
      "gcrf_to_teme_matrix",
      [](double jd_tt) {
        return matrix_to_array(omni_frames::gcrf_to_teme_matrix(jd_tt));
      },
      py::arg("jd_tt"));
  module.def(
      "teme_to_gcrf_matrix",
      [](double jd_tt) {
        return matrix_to_array(omni_frames::teme_to_gcrf_matrix(jd_tt));
      },
      py::arg("jd_tt"));
  module.def(
      "teme_to_ecef_matrix",
      [](double jd_ut1, double xp, double yp) {
        return matrix_to_array(
            omni_frames::teme_to_ecef_matrix(jd_ut1, xp, yp));
      },
      py::arg("jd_ut1"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
  module.def(
      "ecef_to_teme_matrix",
      [](double jd_ut1, double xp, double yp) {
        return matrix_to_array(
            omni_frames::ecef_to_teme_matrix(jd_ut1, xp, yp));
      },
      py::arg("jd_ut1"), py::arg("xp") = 0.0, py::arg("yp") = 0.0);
  module.def("frame_rotation", &frame_rotation, py::arg("source"),
             py::arg("target"), py::kw_only(), py::arg("jd_ut1") = py::none(),
             py::arg("jd_tt") = py::none(), py::arg("xp") = 0.0,
             py::arg("yp") = 0.0, py::arg("position_gcrf") = py::none(),
             py::arg("velocity_gcrf") = py::none(),
             py::arg("q_body_from_gcrf") = py::none());
  module.def("convert_vector", &convert_vector, py::arg("vector"),
             py::arg("source"), py::arg("target"), py::kw_only(),
             py::arg("jd_ut1") = py::none(), py::arg("jd_tt") = py::none(),
             py::arg("xp") = 0.0, py::arg("yp") = 0.0,
             py::arg("position_gcrf") = py::none(),
             py::arg("velocity_gcrf") = py::none(),
             py::arg("q_body_from_gcrf") = py::none());
  module.def(
      "ecef_to_geodetic",
      [](py::object position) {
        return vector_to_array(omni_frames::ecef_to_geodetic(
            to_fixed_vector<3>(position, "position_ecef_m")));
      },
      py::arg("position_ecef_m"));
  module.def(
      "gcrf_position_to_geodetic",
      [](py::object position, double jd_ut1, double jd_tt, double xp,
         double yp) {
        return vector_to_array(omni_frames::gcrf_position_to_geodetic(
            to_fixed_vector<3>(position, "position_gcrf_m"), jd_ut1, jd_tt, xp,
            yp));
      },
      py::arg("position_gcrf_m"), py::arg("jd_ut1"), py::arg("jd_tt"),
      py::arg("xp") = 0.0, py::arg("yp") = 0.0);
}

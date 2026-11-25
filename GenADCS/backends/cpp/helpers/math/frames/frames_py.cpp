#include "bindings_common.hpp"

using namespace frame_bindings;

void bind_conversions(py::module_ &module);

PYBIND11_MODULE(_frames, module) {
  module.doc() = "MISRA-oriented C++ reference-frame transforms";
  module.def(
      "inertial_to_body_matrix",
      [](py::object quaternion) {
        return matrix_to_array(omni_frames::inertial_to_body_matrix(
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
      },
      py::arg("q_b_from_i"));
  module.def(
      "body_to_inertial_matrix",
      [](py::object quaternion) {
        return matrix_to_array(omni_frames::body_to_inertial_matrix(
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
      },
      py::arg("q_b_from_i"));
  module.def(
      "inertial_to_body",
      [](py::object vector, py::object quaternion) {
        return vector_to_array(omni_frames::inertial_to_body(
            to_fixed_vector<3>(vector, "vector_i"),
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
      },
      py::arg("vector_i"), py::arg("q_b_from_i"));
  module.def(
      "body_to_inertial",
      [](py::object vector, py::object quaternion) {
        return vector_to_array(omni_frames::body_to_inertial(
            to_fixed_vector<3>(vector, "vector_b"),
            to_fixed_vector<4>(quaternion, "q_b_from_i")));
      },
      py::arg("vector_b"), py::arg("q_b_from_i"));
  module.def(
      "transform_vector",
      [](py::object vector, py::object matrix) {
        return vector_to_array(omni_frames::transform_vector(
            to_fixed_vector<3>(vector, "vector"),
            to_matrix(matrix, "source_to_target_dcm")));
      },
      py::arg("vector"), py::arg("source_to_target_dcm"));
  module.def(
      "transform_covariance",
      [](py::object covariance, py::object matrix) {
        return matrix_to_array(omni_frames::transform_covariance(
            to_matrix(covariance, "covariance_source"),
            to_matrix(matrix, "source_to_target_dcm")));
      },
      py::arg("covariance_source"), py::arg("source_to_target_dcm"));

  bind_conversions(module);
}

#include "projected_area.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::M;
using omni::dynamics::Q;
using omni::dynamics::V;

void bind_srp_model_projected_area(py::module_ &m)
{
    m.def(
        "projected_area_srp_force",
        [](const V &p, const V &s, const Q &q, const Eigen::MatrixXd &n, const Eigen::VectorXd &a,
           const Eigen::MatrixXd &optical,
           double light) { return omni::dynamics::srp(p, s, q, n, a, optical, Eigen::MatrixXd(), light, false); },
        py::arg("position_i"), py::arg("sun_position_i"), py::arg("q_b_from_i"), py::arg("normals_b"), py::arg("areas"),
        py::arg("optical_coefficients"), py::arg("illumination") = 1.0);
    m.def(
        "projected_area_srp_torque",
        [](const V &p, const V &s, const Q &q, const Eigen::MatrixXd &n, const Eigen::MatrixXd &arms,
           const Eigen::VectorXd &a, const Eigen::MatrixXd &optical,
           double light) { return omni::dynamics::srp(p, s, q, n, a, optical, arms, light, true); },
        py::arg("position_i"), py::arg("sun_position_i"), py::arg("q_b_from_i"), py::arg("normals_b"),
        py::arg("arms_b"), py::arg("areas"), py::arg("optical_coefficients"), py::arg("illumination") = 1.0);
}

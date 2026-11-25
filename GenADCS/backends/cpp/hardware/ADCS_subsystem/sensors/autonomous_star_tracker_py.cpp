#include "../../../helpers/math/rotations/sensor_kernels_py.hpp"
#include "autonomous_star_tracker.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace sensors = omni::hardware::sensors;

namespace {
// Tracker outputs include both 3-vectors and scalar-first 4-vector quaternions.
template <std::size_t N>
py::array_t<double> vector_array(const std::array<double, N> &values) {
  py::array_t<double> result(static_cast<py::ssize_t>(N));
  auto out = result.mutable_unchecked<1>();
  for (std::size_t i = 0; i < N; ++i) {
    out(static_cast<py::ssize_t>(i)) = values[i];
  }
  return result;
}
} // namespace

void bind_sensors_autonomous_star_tracker(py::module_ &module) {
  py::class_<sensors::AutonomousStarTracker>(module, "AutonomousStarTracker")
      .def(py::init([](py::handle p, py::handle o, py::handle m,
                       double saturation, py::handle noise, py::handle bias,
                       double fov) {
             return sensors::AutonomousStarTracker(
                 sensors::kernels::read_vector3(p, "position"),
                 sensors::kernels::read_quaternion(o, "orientation"),
                 sensors::kernels::read_quaternion(m, "misalignment"),
                 saturation, sensors::kernels::read_matrix(noise, "noise"),
                 sensors::kernels::read_vector3(bias, "bias"), fov);
           }),
           py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
           py::arg("saturation_value"), py::arg("noise"), py::arg("bias"),
           py::arg("fov_half_angle") = 3.14159265358979323846)
      .def_property_readonly("position",
                             [](const sensors::AutonomousStarTracker &s) {
                               return vector_array(s.position);
                             })
      .def_property_readonly("orientation",
                             [](const sensors::AutonomousStarTracker &s) {
                               return vector_array(s.orientation);
                             })
      .def_property_readonly("misalignment",
                             [](const sensors::AutonomousStarTracker &s) {
                               return vector_array(s.misalignment);
                             })
      .def_property_readonly("noise",
                             [](const sensors::AutonomousStarTracker &s) {
                               return sensors::kernels::array_matrix(s.noise);
                             })
      .def_property_readonly("noise_cholesky",
                             [](const sensors::AutonomousStarTracker &s) {
                               return sensors::kernels::array_matrix(
                                   s.noise_cholesky);
                             })
      .def_property_readonly("bias",
                             [](const sensors::AutonomousStarTracker &s) {
                               return vector_array(s.bias);
                             })
      .def_readonly("saturation_value",
                    &sensors::AutonomousStarTracker::saturation_value)
      .def_readonly("fov_half_angle",
                    &sensors::AutonomousStarTracker::fov_half_angle)
      .def_readwrite("enabled", &sensors::AutonomousStarTracker::enabled)
      .def_readwrite("seed", &sensors::AutonomousStarTracker::seed)
      .def("noiseless_biasfree_measurement",
           [](const sensors::AutonomousStarTracker &s, py::handle q) {
             return vector_array(s.noiseless_biasfree_measurement(
                 sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noiseless_biased_measurement",
           [](const sensors::AutonomousStarTracker &s, py::handle q) {
             return vector_array(s.noiseless_biased_measurement(
                 sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noisy_biasfree_measurement",
           [](sensors::AutonomousStarTracker &s, py::handle q) {
             return vector_array(s.noisy_biasfree_measurement(
                 sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noisy_biased_measurement",
           [](sensors::AutonomousStarTracker &s, py::handle q) {
             return vector_array(s.noisy_biased_measurement(
                 sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("residual",
           [](const sensors::AutonomousStarTracker &s, py::handle m,
              py::handle e) {
             return vector_array(s.residual(
                 sensors::kernels::read_quaternion(m, "measured_attitude"),
                 sensors::kernels::read_quaternion(e, "estimated_attitude")));
           })
      .def("jacobian_relative_quaternion",
           [](const sensors::AutonomousStarTracker &s) {
             return sensors::kernels::array_matrix(
                 s.jacobian_relative_quaternion());
           })
      .def("jacobian_relative_mrp",
           [](const sensors::AutonomousStarTracker &s, py::handle d) {
             return sensors::kernels::array_matrix(s.jacobian_relative_mrp(
                 sensors::kernels::read_vector3(d, "delta_mrp")));
           })
      .def("jacobian_relative_angle_vector",
           [](const sensors::AutonomousStarTracker &s, py::handle d) {
             return sensors::kernels::array_matrix(
                 s.jacobian_relative_angle_vector(
                     sensors::kernels::read_vector3(d, "delta_angle")));
           })
      .def(
          "availability",
          [](const sensors::AutonomousStarTracker &s, py::object q,
             py::object earth, py::object sun) {
            std::optional<sensors::AutonomousStarTracker::Q> attitude;
            std::optional<sensors::AutonomousStarTracker::V> e, so;
            if (!q.is_none()) {
              attitude = sensors::kernels::read_quaternion(q, "attitude");
            }
            if (!earth.is_none()) {
              e = sensors::kernels::read_vector3(earth, "earth_inertial");
            }
            if (!sun.is_none()) {
              so = sensors::kernels::read_vector3(sun, "sun_inertial");
            }
            return s.availability(attitude, e, so);
          },
          py::arg("attitude") = py::none(),
          py::arg("earth_inertial") = py::none(),
          py::arg("sun_inertial") = py::none())
      .def("fisher_information_matrix",
           [](const sensors::AutonomousStarTracker &s) {
             return sensors::kernels::array_matrix(
                 s.fisher_information_matrix());
           });
}

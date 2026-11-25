#include "../../../helpers/math/rotations/sensor_kernels_py.hpp"
#include "../adcs_subsystem.hpp"
#include "../effects_py.hpp"
#include "../magnetorquer.hpp"
#include "../reaction_wheel.hpp"
#include "accelerometer.hpp"
#include "coarse_sun_sensor.hpp"
#include "fine_earth_sensor.hpp"
#include "fine_sun_sensor.hpp"
#include "gyroscope.hpp"
#include "magnetometer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace hardware = omni::hardware;
namespace native_sensors = omni::hardware::sensors;

namespace {
template <typename Values>
py::array_t<double> vector_array(const Values &values) {
  py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
  auto out = result.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
    out(i) = values[static_cast<std::size_t>(i)];
}
  return result;
}

template <typename Matrix>
py::array_t<double> matrix_array(const Matrix &values) {
  py::array_t<double> result({3, 3});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < 3; ++i) {
    for (py::ssize_t j = 0; j < 3; ++j) {
      out(i, j) =
          values[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}
}
  return result;
}
} // namespace

void bind_sensors_coarse_sun_sensor(py::module_ &module) {
  py::class_<native_sensors::CoarseSunSensor>(module, "CoarseSunSensor")
      .def(py::init([](py::object p, py::object q, py::object m, py::object n,
                       double sensitivity, double saturation, double variance,
                       double bias, double fov) {
             return native_sensors::CoarseSunSensor(
                 native_sensors::kernels::read_vector3(p, "position"),
                 native_sensors::kernels::read_quaternion(q, "orientation"),
                 native_sensors::kernels::read_quaternion(m, "misalignment"),
                 native_sensors::kernels::read_vector3(n, "normal"),
                 sensitivity, saturation, variance, bias, fov);
           }),
           py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
           py::arg("normal"), py::arg("sensitivity"),
           py::arg("saturation_value"), py::arg("noise_variance"),
           py::arg("bias"), py::arg("fov_half_angle"))
      .def_property_readonly("position",
                             [](const native_sensors::CoarseSunSensor &s) {
                               return vector_array(s.position);
                             })
      .def_property_readonly("orientation",
                             [](const native_sensors::CoarseSunSensor &s) {
                               return vector_array(s.orientation);
                             })
      .def_property_readonly("misalignment",
                             [](const native_sensors::CoarseSunSensor &s) {
                               return vector_array(s.misalignment);
                             })
      .def_property_readonly("normal",
                             [](const native_sensors::CoarseSunSensor &s) {
                               return vector_array(s.normal);
                             })
      .def_readonly("sensitivity",
                    &native_sensors::CoarseSunSensor::sensitivity)
      .def_readonly("saturation_value",
                    &native_sensors::CoarseSunSensor::saturation_value)
      .def_readonly("noise_variance",
                    &native_sensors::CoarseSunSensor::noise_variance)
      .def_readonly("noise_std", &native_sensors::CoarseSunSensor::noise_std)
      .def_readonly("bias", &native_sensors::CoarseSunSensor::bias)
      .def_readonly("fov_half_angle",
                    &native_sensors::CoarseSunSensor::fov_half_angle)
      .def_readwrite("enabled", &native_sensors::CoarseSunSensor::enabled)
      .def_readwrite("seed", &native_sensors::CoarseSunSensor::seed)
      .def("noiseless_biasfree_measurement",
           [](const native_sensors::CoarseSunSensor &s, py::handle sun,
              py::handle q) {
             return s.noiseless_biasfree_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"));
           })
      .def("noiseless_biased_measurement",
           [](const native_sensors::CoarseSunSensor &s, py::handle sun,
              py::handle q) {
             return s.noiseless_biased_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"));
           })
      .def(
          "noisy_biasfree_measurement",
          [](native_sensors::CoarseSunSensor &s, py::handle sun, py::handle q) {
            return s.noisy_biasfree_measurement(
                native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                native_sensors::kernels::read_quaternion(q, "attitude"));
          })
      .def(
          "noisy_biased_measurement",
          [](native_sensors::CoarseSunSensor &s, py::handle sun, py::handle q) {
            return s.noisy_biased_measurement(
                native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                native_sensors::kernels::read_quaternion(q, "attitude"));
          })
      .def("jacobian_bias", &native_sensors::CoarseSunSensor::jacobian_bias)
      .def("jacobian_relative_quaternion",
           [](const native_sensors::CoarseSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return vector_array(s.jacobian_relative_quaternion(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_quaternion(d,
                                                          "delta_quaternion")));
           })
      .def("jacobian_relative_mrp",
           [](const native_sensors::CoarseSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return vector_array(s.jacobian_relative_mrp(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_vector3(d, "delta_mrp")));
           })
      .def("jacobian_relative_angle_vector",
           [](const native_sensors::CoarseSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return vector_array(s.jacobian_relative_angle_vector(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_vector3(d, "delta_angle")));
           })
      .def(
          "availability",
          [](const native_sensors::CoarseSunSensor &s, py::handle sun,
             py::handle q, bool visible) {
            return s.availability(
                native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                native_sensors::kernels::read_quaternion(q, "attitude"),
                visible);
          },
          py::arg("sun_inertial"), py::arg("attitude"),
          py::arg("sun_visible") = true)
      .def(
          "fisher_information_matrix",
          [](const native_sensors::CoarseSunSensor &s, py::handle sun,
             py::handle q, bool visible) {
            return matrix_array(s.fisher_information_matrix(
                native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                native_sensors::kernels::read_quaternion(q, "attitude"),
                visible));
          },
          py::arg("sun_inertial"), py::arg("attitude"),
          py::arg("sun_visible") = true);
}

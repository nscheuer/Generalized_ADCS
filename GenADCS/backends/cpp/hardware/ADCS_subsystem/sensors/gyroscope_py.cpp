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

void bind_sensors_gyroscope(py::module_ &module) {
  py::class_<native_sensors::Gyroscope>(module, "Gyroscope")
      .def(py::init([](py::object position, py::object orientation,
                       py::object misalignment, double saturation_value,
                       py::object noise, py::object bias) {
             return native_sensors::Gyroscope(
                 native_sensors::kernels::read_vector3(position, "position"),
                 native_sensors::kernels::read_quaternion(orientation,
                                                          "orientation"),
                 native_sensors::kernels::read_quaternion(misalignment,
                                                          "misalignment"),
                 saturation_value,
                 native_sensors::kernels::read_matrix(noise, "noise"),
                 native_sensors::kernels::read_vector3(bias, "bias"));
           }),
           py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
           py::arg("saturation_value"), py::arg("noise"), py::arg("bias"))
      .def_property_readonly("position",
                             [](const native_sensors::Gyroscope &s) {
                               return vector_array(s.position);
                             })
      .def_property_readonly("orientation",
                             [](const native_sensors::Gyroscope &s) {
                               return vector_array(s.orientation);
                             })
      .def_property_readonly("misalignment",
                             [](const native_sensors::Gyroscope &s) {
                               return vector_array(s.misalignment);
                             })
      .def_readonly("saturation_value",
                    &native_sensors::Gyroscope::saturation_value)
      .def_property_readonly("noise",
                             [](const native_sensors::Gyroscope &s) {
                               return matrix_array(s.noise);
                             })
      .def_property_readonly("noise_cholesky",
                             [](const native_sensors::Gyroscope &s) {
                               return matrix_array(s.noise_cholesky);
                             })
      .def_property_readonly("bias",
                             [](const native_sensors::Gyroscope &s) {
                               return vector_array(s.bias);
                             })
      .def_readwrite("enabled", &native_sensors::Gyroscope::enabled)
      .def_readwrite("seed", &native_sensors::Gyroscope::seed)
      .def("noiseless_biasfree_measurement",
           [](const native_sensors::Gyroscope &s, py::object rate) {
             return native_sensors::kernels::array_vector(
                 s.noiseless_biasfree_measurement(
                     native_sensors::kernels::read_vector3(
                         rate, "angular_velocity_body")));
           })
      .def("noiseless_biased_measurement",
           [](const native_sensors::Gyroscope &s, py::object rate) {
             return native_sensors::kernels::array_vector(
                 s.noiseless_biased_measurement(
                     native_sensors::kernels::read_vector3(
                         rate, "angular_velocity_body")));
           })
      .def("noisy_biasfree_measurement",
           [](native_sensors::Gyroscope &s, py::object rate) {
             return native_sensors::kernels::array_vector(
                 s.noisy_biasfree_measurement(
                     native_sensors::kernels::read_vector3(
                         rate, "angular_velocity_body")));
           })
      .def("noisy_biased_measurement",
           [](native_sensors::Gyroscope &s, py::object rate) {
             return native_sensors::kernels::array_vector(
                 s.noisy_biased_measurement(
                     native_sensors::kernels::read_vector3(
                         rate, "angular_velocity_body")));
           })
      .def("jacobian_angular_velocity",
           [](const native_sensors::Gyroscope &s) {
             return matrix_array(s.jacobian_angular_velocity());
           })
      .def("jacobian_bias",
           [](const native_sensors::Gyroscope &s) {
             return matrix_array(s.jacobian_bias());
           })
      .def("availability", &native_sensors::Gyroscope::availability);
}

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

template <std::size_t N>
std::array<double, N> read_fixed(py::handle value, const char *name) {
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!array || array.ndim() != 1 ||
      array.shape(0) != static_cast<py::ssize_t>(N)) {
    throw py::value_error(std::string(name) + " has an invalid shape");
}
  std::array<double, N> result{};
  auto input = array.unchecked<1>();
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = input(static_cast<py::ssize_t>(i));
}
  return result;
}
} // namespace

void bind_sensors_fine_earth_sensor(py::module_ &module) {
  py::class_<native_sensors::FineEarthSensor, native_sensors::FineSunSensor>(
      module, "FineEarthSensor")
      .def(
          py::init([](py::object p, py::object q, py::object m, py::object n,
                      py::object b, py::object f) {
            native_sensors::FineSunSensor::M2 covariance{};
            auto a = py::array_t<double, py::array::c_style |
                                             py::array::forcecast>::ensure(n);
            if (!a || a.ndim() != 2 || a.shape(0) != 2 || a.shape(1) != 2) {
              throw py::value_error("noise must be 2x2");
}
            auto values = a.unchecked<2>();
            for (int i = 0; i < 2; ++i) {
              for (int j = 0; j < 2; ++j) {
                covariance[i][j] = values(i, j);
}
}
            return native_sensors::FineEarthSensor(
                native_sensors::kernels::read_vector3(p, "position"),
                native_sensors::kernels::normalize(
                    native_sensors::kernels::read_quaternion(q, "orientation")),
                native_sensors::kernels::normalize(
                    native_sensors::kernels::read_quaternion(m,
                                                             "misalignment")),
                covariance, read_fixed<2>(b, "bias"),
                read_fixed<2>(f, "fov_half_angles"));
          }),
          py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
          py::arg("noise"), py::arg("bias"), py::arg("fov_half_angles"))
      .def(
          "availability",
          [](const native_sensors::FineEarthSensor &s, py::handle earth,
             py::handle attitude, bool visible) {
            return s.availability(
                native_sensors::kernels::read_vector3(earth, "earth_inertial"),
                native_sensors::kernels::read_quaternion(attitude, "attitude"),
                visible);
          },
          py::arg("earth_inertial"), py::arg("attitude"),
          py::arg("earth_visible") = true)
      .def(
          "fisher_information_matrix",
          [](const native_sensors::FineEarthSensor &s, py::handle earth,
             py::handle attitude, bool visible) {
            return matrix_array(s.fisher_information_matrix(
                native_sensors::kernels::read_vector3(earth, "earth_inertial"),
                native_sensors::kernels::read_quaternion(attitude, "attitude"),
                visible));
          },
          py::arg("earth_inertial"), py::arg("attitude"),
          py::arg("earth_visible") = true);
}

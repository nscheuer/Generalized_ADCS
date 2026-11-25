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

template <typename Matrix>
py::array_t<double> matrix_array_2x2(const Matrix &values) {
  py::array_t<double> result({2, 2});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < 2; ++i) {
    for (py::ssize_t j = 0; j < 2; ++j) {
      out(i, j) = values[i][j];
    }
  }
  return result;
}

template <typename Matrix>
py::array_t<double> matrix_array_2x3(const Matrix &values) {
  py::array_t<double> result({2, 3});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < 2; ++i) {
    for (py::ssize_t j = 0; j < 3; ++j) {
      out(i, j) = values[i][j];
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

template <typename Values>
py::array_t<double> vector_array_2(const Values &values) {
  return vector_array(values);
}
} // namespace

void bind_sensors_fine_sun_sensor(py::module_ &module) {
  py::class_<native_sensors::FineSunSensor>(module, "FineSunSensor")
      .def(
          py::init([](py::handle p, py::handle q, py::handle m, py::handle n,
                      py::handle b, py::handle f) {
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
            const auto bias = read_fixed<2>(b, "bias");
            const auto fov = read_fixed<2>(f, "fov_half_angles");
            return native_sensors::FineSunSensor(
                native_sensors::kernels::read_vector3(p, "position"),
                native_sensors::kernels::normalize(
                    native_sensors::kernels::read_quaternion(q, "orientation")),
                native_sensors::kernels::normalize(
                    native_sensors::kernels::read_quaternion(m,
                                                             "misalignment")),
                covariance, bias, fov);
          }),
          py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
          py::arg("noise"), py::arg("bias"), py::arg("fov_half_angles"))
      .def_property_readonly("position",
                             [](const native_sensors::FineSunSensor &s) {
                               return vector_array(s.position);
                             })
      .def_property_readonly("orientation",
                             [](const native_sensors::FineSunSensor &s) {
                               return vector_array(s.orientation);
                             })
      .def_property_readonly("misalignment",
                             [](const native_sensors::FineSunSensor &s) {
                               return vector_array(s.misalignment);
                             })
      .def_readwrite("enabled", &native_sensors::FineSunSensor::enabled)
      .def_readwrite("seed", &native_sensors::FineSunSensor::seed)
      .def_readonly("noise", &native_sensors::FineSunSensor::noise)
      .def_readonly("bias", &native_sensors::FineSunSensor::bias)
      .def_readonly("fov_half_angles",
                    &native_sensors::FineSunSensor::fov_half_angles)
      .def("noiseless_biasfree_measurement",
           [](const native_sensors::FineSunSensor &s, py::handle sun,
              py::handle q) {
             return vector_array_2(s.noiseless_biasfree_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noiseless_biased_measurement",
           [](const native_sensors::FineSunSensor &s, py::handle sun,
              py::handle q) {
             return vector_array_2(s.noiseless_biased_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noisy_biasfree_measurement",
           [](native_sensors::FineSunSensor &s, py::handle sun, py::handle q) {
             return vector_array_2(s.noisy_biasfree_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("noisy_biased_measurement",
           [](native_sensors::FineSunSensor &s, py::handle sun, py::handle q) {
             return vector_array_2(s.noisy_biased_measurement(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude")));
           })
      .def("jacobian_bias",
           [](const native_sensors::FineSunSensor &s) {
             return matrix_array_2x2(s.jacobian_bias());
           })
      .def("jacobian_relative_quaternion",
           [](const native_sensors::FineSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return matrix_array_2x3(s.jacobian_relative_quaternion(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_quaternion(d,
                                                          "delta_quaternion")));
           })
      .def("jacobian_relative_mrp",
           [](const native_sensors::FineSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return matrix_array_2x3(s.jacobian_relative_mrp(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_vector3(d, "delta_mrp")));
           })
      .def("jacobian_relative_angle_vector",
           [](const native_sensors::FineSunSensor &s, py::handle sun,
              py::handle q, py::handle d) {
             return matrix_array_2x3(s.jacobian_relative_angle_vector(
                 native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                 native_sensors::kernels::read_quaternion(q, "attitude"),
                 native_sensors::kernels::read_vector3(d, "delta_angle")));
           })
      .def(
          "availability",
          [](const native_sensors::FineSunSensor &s, py::handle sun,
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
          [](const native_sensors::FineSunSensor &s, py::handle sun,
             py::handle q, bool visible) {
            return matrix_array(s.fisher_information_matrix(
                native_sensors::kernels::read_vector3(sun, "sun_inertial"),
                native_sensors::kernels::read_quaternion(q, "attitude"),
                visible));
          },
          py::arg("sun_inertial"), py::arg("attitude"),
          py::arg("sun_visible") = true);
}

#include "gnss.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace {
template <std::size_t N>
std::array<double, N> read_vector(py::handle obj, const char *name) {
  auto a =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          obj);
  if (!a || a.ndim() != 1 || a.shape(0) != static_cast<py::ssize_t>(N))
    throw py::value_error(std::string(name) + " must have " +
                          std::to_string(N) + " components");
  std::array<double, N> result{};
  auto values = a.unchecked<1>();
  for (std::size_t i = 0; i < N; ++i)
    result[i] = values(static_cast<py::ssize_t>(i));
  return result;
}
py::array_t<double> array63(const omni::hardware::sensors::GNSS::M63 &x) {
  py::array_t<double> out({6, 3});
  auto a = out.mutable_unchecked<2>();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j)
      a(i, j) = x[i][j];
  return out;
}
py::array_t<double> array6(const omni::hardware::sensors::GNSS::M6 &x) {
  py::array_t<double> out({6, 6});
  auto a = out.mutable_unchecked<2>();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      a(i, j) = x[i][j];
  return out;
}
py::array_t<double> arrayv(const omni::hardware::sensors::GNSS::V6 &x) {
  py::array_t<double> out(6);
  auto a = out.mutable_unchecked<1>();
  for (int i = 0; i < 6; ++i)
    a(i) = x[i];
  return out;
}
} // namespace

void bind_sensors_gnss(py::module_ &module) {
  using S = omni::hardware::sensors::GNSS;
  py::class_<S>(module, "GNSS")
      .def(py::init([](py::handle noise, py::handle bias) {
             auto n =
                 py::array_t<double, py::array::c_style |
                                         py::array::forcecast>::ensure(noise);
             if (!n || n.ndim() != 2 || n.shape(0) != 6 || n.shape(1) != 6)
               throw py::value_error("noise must have shape (6, 6)");
             S::M6 covariance{};
             auto values = n.unchecked<2>();
             for (int i = 0; i < 6; ++i)
               for (int j = 0; j < 6; ++j)
                 covariance[i][j] = values(i, j);
             return S(covariance, read_vector<6>(bias, "bias"));
           }),
           py::arg("noise"), py::arg("bias"))
      .def_readonly("noise", &S::noise)
      .def_readonly("noise_cholesky", &S::noise_cholesky)
      .def_readonly("bias", &S::bias)
      .def_readwrite("enabled", &S::enabled)
      .def_readwrite("seed", &S::seed)
      .def_readwrite("random_tick", &S::random_tick)
      .def("noiseless_biasfree_measurement",
           [](const S &s, py::handle p, py::handle v) {
             return arrayv(s.noiseless_biasfree_measurement(
                 read_vector<3>(p, "position_inertial"),
                 read_vector<3>(v, "velocity_inertial")));
           })
      .def("noiseless_biased_measurement",
           [](const S &s, py::handle p, py::handle v) {
             return arrayv(s.noiseless_biased_measurement(
                 read_vector<3>(p, "position_inertial"),
                 read_vector<3>(v, "velocity_inertial")));
           })
      .def("noisy_biasfree_measurement",
           [](S &s, py::handle p, py::handle v) {
             return arrayv(s.noisy_biasfree_measurement(
                 read_vector<3>(p, "position_inertial"),
                 read_vector<3>(v, "velocity_inertial")));
           })
      .def("noisy_biased_measurement",
           [](S &s, py::handle p, py::handle v) {
             return arrayv(s.noisy_biased_measurement(
                 read_vector<3>(p, "position_inertial"),
                 read_vector<3>(v, "velocity_inertial")));
           })
      .def("jacobian_relative_quaternion",
           [](const S &s) { return array63(s.jacobian_relative_quaternion()); })
      .def("jacobian_relative_mrp",
           [](const S &s) { return array63(s.jacobian_relative_mrp()); })
      .def("jacobian_relative_angle_vector",
           [](const S &s) {
             return array63(s.jacobian_relative_angle_vector());
           })
      .def("jacobian_position",
           [](const S &s) { return array63(s.jacobian_position()); })
      .def("jacobian_velocity",
           [](const S &s) { return array63(s.jacobian_velocity()); })
      .def("jacobian_bias",
           [](const S &s) { return array6(s.jacobian_bias()); })
      .def("availability", &S::availability);
}

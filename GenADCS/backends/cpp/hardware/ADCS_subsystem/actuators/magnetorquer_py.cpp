#include "magnetorquer.hpp"

#include <cmath>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
struct BoundMagnetorquer {
  hardware::Magnetorquer native;
  std::optional<std::string> name;

  explicit BoundMagnetorquer(hardware::Magnetorquer value,
                             std::optional<std::string> n = std::nullopt)
      : native(std::move(value)), name(std::move(n)) {}
};

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

void bind_magnetorquer(py::module_ &module) {
  py::class_<BoundMagnetorquer>(module, "Magnetorquer")
      .def(py::init([](py::object position, py::object orientation,
                       double max_dipole, py::object misalignment,
                       double noise, double bias,
                       std::optional<std::string> name) {
             if (misalignment.is_none()) {
               misalignment = py::make_tuple(0.0, 0.0, 0.0);
             }
             return BoundMagnetorquer(
                 hardware::Magnetorquer(
                     read_fixed<3>(position, "position"),
                     read_fixed<3>(orientation, "orientation"), max_dipole,
                     read_fixed<3>(misalignment, "misalignment"), noise, bias),
                 std::move(name));
           }),
           py::arg("position"), py::arg("orientation"), py::arg("max_dipole"),
           py::arg("misalignment") = py::none(), py::arg("noise") = 0.0,
           py::arg("bias") = 0.0, py::arg("name") = std::nullopt)
      .def_readwrite("name", &BoundMagnetorquer::name)
      .def_property(
          "position",
          [](const BoundMagnetorquer &self) {
            return py::make_tuple(self.native.position[0],
                                  self.native.position[1],
                                  self.native.position[2]);
          },
          [](BoundMagnetorquer &self, py::handle value) {
            self.native.position = read_fixed<3>(value, "position");
          })
      .def_property(
          "orientation",
          [](const BoundMagnetorquer &self) {
            return py::make_tuple(self.native.orientation[0],
                                  self.native.orientation[1],
                                  self.native.orientation[2]);
          },
          [](BoundMagnetorquer &self, py::handle value) {
            self.native.orientation = read_fixed<3>(value, "orientation");
          })
      .def_property(
          "max_dipole",
          [](const BoundMagnetorquer &self) { return self.native.max_dipole; },
          [](BoundMagnetorquer &self, double value) {
            self.native.max_dipole = value;
          })
      .def_property(
          "misalignment",
          [](const BoundMagnetorquer &self) {
            return py::make_tuple(self.native.misalignment[0],
                                  self.native.misalignment[1],
                                  self.native.misalignment[2]);
          },
          [](BoundMagnetorquer &self, py::handle value) {
            self.native.misalignment = read_fixed<3>(value, "misalignment");
          })
      .def_property(
          "noise",
          [](const BoundMagnetorquer &self) { return self.native.noise; },
          [](BoundMagnetorquer &self, double value) {
            if (!std::isfinite(value) || value < 0.0) {
              throw py::value_error("noise must be finite and nonnegative");
            }
            self.native.noise = value;
          })
      .def_property(
          "bias",
          [](const BoundMagnetorquer &self) { return self.native.bias; },
          [](BoundMagnetorquer &self, double value) {
            if (!std::isfinite(value)) {
              throw py::value_error("bias must be finite");
            }
            self.native.bias = value;
          })
      .def("__eq__",
           [](const BoundMagnetorquer &left, const BoundMagnetorquer &right) {
             return left.native.position == right.native.position &&
                    left.native.orientation == right.native.orientation &&
                    (left.native.max_dipole == right.native.max_dipole) &&
                    left.native.misalignment == right.native.misalignment &&
                    (left.native.noise == right.native.noise) &&
                    (left.native.bias == right.native.bias);
           });
}

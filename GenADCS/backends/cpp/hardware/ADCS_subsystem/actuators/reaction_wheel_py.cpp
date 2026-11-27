#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"

#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
struct BoundReactionWheel {
  hardware::ReactionWheel native;
  py::object position, orientation, misalignment;

  BoundReactionWheel(hardware::ReactionWheel value, py::object p, py::object o,
                     py::object m)
      : native(std::move(value)), position(std::move(p)),
        orientation(std::move(o)), misalignment(std::move(m)) {}
};

bool same(const py::object &left, const py::object &right) {
  return py::module_::import("numpy")
      .attr("array_equal")(left, right)
      .cast<bool>();
}
} // namespace

void bind_reaction_wheel(py::module_ &module) {
  py::class_<BoundReactionWheel>(module, "ReactionWheel")
      .def(py::init([](py::handle position, py::handle orientation,
                       double rotational_inertia, double max_torque,
                       double max_momentum, py::object misalignment,
                       double noise, double bias) {
             if (misalignment.is_none()) {
               misalignment = py::make_tuple(0.0, 0.0, 0.0);
             }
             auto native = hardware::ReactionWheel(
                 {}, {}, rotational_inertia, max_torque, max_momentum, {},
                 noise, bias);
             return BoundReactionWheel(
                 std::move(native),
                 py::reinterpret_borrow<py::object>(position),
                 py::reinterpret_borrow<py::object>(orientation), misalignment);
           }),
           py::arg("position"), py::arg("orientation"),
           py::arg("rotational_inertia"), py::arg("max_torque"),
           py::arg("max_momentum"), py::arg("misalignment") = py::none(),
           py::arg("noise") = 0.0, py::arg("bias") = 0.0)
      .def_readwrite("position", &BoundReactionWheel::position)
      .def_readwrite("orientation", &BoundReactionWheel::orientation)
      .def_property(
          "rotational_inertia",
          [](const BoundReactionWheel &self) {
            return self.native.rotational_inertia;
          },
          [](BoundReactionWheel &self, double value) {
            self.native.rotational_inertia = value;
          })
      .def_property(
          "max_torque",
          [](const BoundReactionWheel &self) { return self.native.max_torque; },
          [](BoundReactionWheel &self, double value) {
            self.native.max_torque = value;
          })
      .def_property(
          "max_momentum",
          [](const BoundReactionWheel &self) {
            return self.native.max_momentum;
          },
          [](BoundReactionWheel &self, double value) {
            self.native.max_momentum = value;
          })
      .def_readwrite("misalignment", &BoundReactionWheel::misalignment)
      .def_property(
          "noise",
          [](const BoundReactionWheel &self) { return self.native.noise; },
          [](BoundReactionWheel &self, double value) {
            if (!std::isfinite(value) || value < 0.0) {
              throw py::value_error("noise must be finite and nonnegative");
            }
            self.native.noise = value;
          })
      .def_property(
          "bias",
          [](const BoundReactionWheel &self) { return self.native.bias; },
          [](BoundReactionWheel &self, double value) {
            if (!std::isfinite(value)) {
              throw py::value_error("bias must be finite");
            }
            self.native.bias = value;
          })
      .def("__eq__",
           [](const BoundReactionWheel &left, const BoundReactionWheel &right) {
             return same(left.position, right.position) &&
                    same(left.orientation, right.orientation) &&
                    (left.native.rotational_inertia ==
                     right.native.rotational_inertia) &&
                    (left.native.max_torque == right.native.max_torque) &&
                    (left.native.max_momentum == right.native.max_momentum) &&
                    same(left.misalignment, right.misalignment) &&
                    (left.native.noise == right.native.noise) &&
                    (left.native.bias == right.native.bias);
           });
}

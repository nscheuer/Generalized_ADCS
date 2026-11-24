#include "adcs_subsystem.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
struct BoundADCSSubsystem {
    hardware::ADCSSubsystem native;
    py::list reaction_wheels;
    py::list magnetorquers;

    BoundADCSSubsystem(py::list wheels, py::list torquers)
        : native(), reaction_wheels(std::move(wheels)),
          magnetorquers(std::move(torquers)) {}
};

py::list as_list(const py::object& value) {
    if (value.is_none()) return py::list();
    if (PyList_Check(value.ptr()) != 0) return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

std::vector<hardware::ReactionWheel> native_wheels(const py::list& wheels) {
    std::vector<hardware::ReactionWheel> result;
    result.reserve(wheels.size());
    for (const py::handle wheel : wheels) {
        const auto position = py::cast<std::array<double, 3>>(wheel.attr("position"));
        const auto orientation = py::cast<std::array<double, 3>>(wheel.attr("orientation"));
        const auto misalignment = py::cast<std::array<double, 3>>(wheel.attr("misalignment"));
        result.emplace_back(position, orientation,
            py::cast<double>(wheel.attr("rotational_inertia")),
            py::cast<double>(wheel.attr("max_torque")),
            py::cast<double>(wheel.attr("max_momentum")), misalignment);
    }
    return result;
}

std::vector<hardware::Magnetorquer> native_magnetorquers(const py::list& torquers) {
    std::vector<hardware::Magnetorquer> result;
    result.reserve(torquers.size());
    for (const py::handle torquer : torquers) {
        const auto position = py::cast<std::array<double, 3>>(torquer.attr("position"));
        const auto orientation = py::cast<std::array<double, 3>>(torquer.attr("orientation"));
        const auto misalignment = py::cast<std::array<double, 3>>(torquer.attr("misalignment"));
        result.emplace_back(position, orientation,
            py::cast<double>(torquer.attr("max_dipole")), misalignment);
    }
    return result;
}

BoundADCSSubsystem make_subsystem(py::object wheels, py::object torquers) {
    auto wheel_list = as_list(wheels);
    auto torquer_list = as_list(torquers);
    BoundADCSSubsystem result(std::move(wheel_list), std::move(torquer_list));
    result.native = hardware::ADCSSubsystem(
        native_wheels(result.reaction_wheels),
        native_magnetorquers(result.magnetorquers));
    return result;
}

bool same(const py::object& left, const py::object& right) {
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}
}

void bind_adcs_subsystem(py::module_& module) {
    py::class_<BoundADCSSubsystem>(module, "ADCSSubsystem")
        .def(py::init([](py::object wheels, py::object torquers) {
            return make_subsystem(std::move(wheels), std::move(torquers));
        }), py::arg("reaction_wheels") = py::none(),
            py::arg("magnetorquers") = py::none())
        .def_property("reaction_wheels",
            [](BoundADCSSubsystem& self) {
                self.native.reaction_wheels = native_wheels(self.reaction_wheels);
                return self.reaction_wheels;
            },
            [](BoundADCSSubsystem& self, py::object value) {
                auto devices = as_list(value);
                auto native = native_wheels(devices);
                self.native.reaction_wheels = std::move(native);
                self.reaction_wheels = std::move(devices);
            })
        .def_property("magnetorquers",
            [](BoundADCSSubsystem& self) {
                self.native.magnetorquers = native_magnetorquers(self.magnetorquers);
                return self.magnetorquers;
            },
            [](BoundADCSSubsystem& self, py::object value) {
                auto devices = as_list(value);
                auto native = native_magnetorquers(devices);
                self.native.magnetorquers = std::move(native);
                self.magnetorquers = std::move(devices);
            })
        .def("__eq__", [](const BoundADCSSubsystem& left,
                           const BoundADCSSubsystem& right) {
            return same(left.reaction_wheels, right.reaction_wheels)
                && same(left.magnetorquers, right.magnetorquers);
        });
}

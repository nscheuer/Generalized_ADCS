#include "adcs_subsystem.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace
{
struct BoundADCSSubsystem
{
    hardware::ADCSSubsystem native;
    py::list reaction_wheels;
    py::list magnetorquers;

    BoundADCSSubsystem(py::list wheels, py::list torquers)
        : native(), reaction_wheels(std::move(wheels)), magnetorquers(std::move(torquers))
    {
    }
};

py::list as_list(const py::object &value)
{
    if (value.is_none())
        return py::list();
    if (PyList_Check(value.ptr()) != 0)
        return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

BoundADCSSubsystem make_subsystem(py::object wheels, py::object torquers)
{
    auto wheel_list = as_list(wheels);
    auto torquer_list = as_list(torquers);
    BoundADCSSubsystem result(std::move(wheel_list), std::move(torquer_list));
    return result;
}

bool same(const py::object &left, const py::object &right)
{
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}
} // namespace

void bind_adcs_subsystem(py::module_ &module)
{
    py::class_<BoundADCSSubsystem>(module, "ADCSSubsystem")
        .def(py::init([](py::object wheels, py::object torquers) {
                 return make_subsystem(std::move(wheels), std::move(torquers));
             }),
             py::arg("reaction_wheels") = py::none(), py::arg("magnetorquers") = py::none())
        .def_property(
            "reaction_wheels", [](BoundADCSSubsystem &self) { return self.reaction_wheels; },
            [](BoundADCSSubsystem &self, py::object value) {
                auto devices = as_list(value);
                self.reaction_wheels = std::move(devices);
            })
        .def_property(
            "magnetorquers", [](BoundADCSSubsystem &self) { return self.magnetorquers; },
            [](BoundADCSSubsystem &self, py::object value) {
                auto devices = as_list(value);
                self.magnetorquers = std::move(devices);
            })
        .def("__eq__", [](const BoundADCSSubsystem &left, const BoundADCSSubsystem &right) {
            return same(left.reaction_wheels, right.reaction_wheels) && same(left.magnetorquers, right.magnetorquers);
        });
}

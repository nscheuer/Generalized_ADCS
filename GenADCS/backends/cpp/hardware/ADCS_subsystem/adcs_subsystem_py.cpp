#include "adcs_subsystem.hpp"
#include "effects.hpp"
#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"
#include "sensors/gyroscope.hpp"
#include "sensors/magnetometer.hpp"
#include "sensors/accelerometer.hpp"
#include "sensors/coarse_sun_sensor.hpp"
#include "sensors/fine_sun_sensor.hpp"
#include "sensors/fine_earth_sensor.hpp"
#include "../../helpers/math/rotations/sensor_kernels_py.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace hardware = omni::hardware;
namespace native_sensors = omni::hardware::sensors;

namespace {
py::list fresh_list(const py::object& value) {
    if (value.is_none()) return py::list();
    if (PyList_Check(value.ptr()) != 0) return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

bool same(const py::object& left, const py::object& right) {
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}

}

void bind_adcs_subsystem(py::module_& module) {
    py::class_<hardware::ADCSSubsystem>(module, "ADCSSubsystem")
        .def(py::init([](py::object reaction_wheels, py::object magnetorquers) {
            return hardware::ADCSSubsystem(fresh_list(reaction_wheels), fresh_list(magnetorquers));
        }), py::arg("reaction_wheels") = py::none(), py::arg("magnetorquers") = py::none())
        .def_readwrite("reaction_wheels", &hardware::ADCSSubsystem::reaction_wheels)
        .def_readwrite("magnetorquers", &hardware::ADCSSubsystem::magnetorquers)
        .def("__eq__", [](const hardware::ADCSSubsystem& left,
                           const hardware::ADCSSubsystem& right) {
            return same(left.reaction_wheels, right.reaction_wheels)
                && same(left.magnetorquers, right.magnetorquers);
        });

}

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
    py::list sensors;
    py::object estimator = py::none();

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

// Construct the variant directly: pybind11's generic variant caster requires
// a default constructor, but the sensor classes require configuration.
hardware::Sensor as_sensor(py::handle value)
{
    namespace sensors = hardware::sensors;
    // Check the derived sensor before its base to preserve its concrete type.
    if (py::isinstance<sensors::FineEarthSensor>(value))
        return hardware::Sensor(std::in_place_type<sensors::FineEarthSensor>, value.cast<const sensors::FineEarthSensor&>());
    if (py::isinstance<sensors::FineSunSensor>(value))
        return hardware::Sensor(std::in_place_type<sensors::FineSunSensor>, value.cast<const sensors::FineSunSensor&>());
    if (py::isinstance<sensors::Gyroscope>(value))
        return hardware::Sensor(std::in_place_type<sensors::Gyroscope>, value.cast<const sensors::Gyroscope&>());
    if (py::isinstance<sensors::Magnetometer>(value))
        return hardware::Sensor(std::in_place_type<sensors::Magnetometer>, value.cast<const sensors::Magnetometer&>());
    if (py::isinstance<sensors::Accelerometer>(value))
        return hardware::Sensor(std::in_place_type<sensors::Accelerometer>, value.cast<const sensors::Accelerometer&>());
    if (py::isinstance<sensors::CoarseSunSensor>(value))
        return hardware::Sensor(std::in_place_type<sensors::CoarseSunSensor>, value.cast<const sensors::CoarseSunSensor&>());
    if (py::isinstance<sensors::GNSS>(value))
        return hardware::Sensor(std::in_place_type<sensors::GNSS>, value.cast<const sensors::GNSS&>());
    if (py::isinstance<sensors::AutonomousStarTracker>(value))
        return hardware::Sensor(std::in_place_type<sensors::AutonomousStarTracker>, value.cast<const sensors::AutonomousStarTracker&>());
    throw py::type_error("sensors must contain native C++ sensor objects");
}

std::vector<hardware::Sensor> as_native_sensors(const py::list &values)
{
    std::vector<hardware::Sensor> result;
    result.reserve(values.size());
    for (py::handle value : values)
        result.push_back(as_sensor(value));
    return result;
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
        .def(py::init([](py::object wheels, py::object torquers, py::object sensors, py::object N, py::object estimator) {
                 if (py::isinstance<py::bool_>(N) || !py::isinstance<py::int_>(N) || N.cast<int>() < 1)
                     throw py::value_error("N must be a positive integer");
                 auto result = make_subsystem(std::move(wheels), std::move(torquers));
                 result.native.N = N.cast<int>();
                 result.sensors = as_list(sensors);
                 result.native.sensors = as_native_sensors(result.sensors);
                 if (!estimator.is_none()) {
                     auto type = py::module_::import("GenADCS.backends.cpp.hardware.ADCS_subsystem.estimator.attitude_estimator").attr("AttitudeEstimator");
                     if (!py::isinstance(estimator, type)) throw py::type_error("estimator must be an AttitudeEstimator");
                 }
                 result.estimator = estimator;
                 return result;
             }),
             py::arg("reaction_wheels") = py::none(), py::arg("magnetorquers") = py::none(), py::arg("sensors") = py::none(), py::arg("N") = 1, py::arg("estimator") = py::none())
        .def_property(
            "sensors", [](BoundADCSSubsystem &self) { return self.sensors; },
            [](BoundADCSSubsystem &self, py::object value) {
                auto devices = as_list(value);
                auto native_devices = as_native_sensors(devices);
                self.native.sensors = std::move(native_devices);
                self.sensors = std::move(devices);
            })
        .def_readwrite("estimator", &BoundADCSSubsystem::estimator)
        .def_property("N", [](const BoundADCSSubsystem& self) { return self.native.N; },
            [](BoundADCSSubsystem& self, py::object n) {
                if (py::isinstance<py::bool_>(n) || !py::isinstance<py::int_>(n) || n.cast<int>() < 1) throw py::value_error("N must be a positive integer");
                self.native.N=n.cast<int>();
            })
        .def("every_step", [](const BoundADCSSubsystem& self, std::uint64_t tick) { return self.native.every_step(tick); })
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

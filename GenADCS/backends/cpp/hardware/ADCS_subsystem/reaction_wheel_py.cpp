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
struct BoundReactionWheel {
    hardware::ReactionWheel native;
    py::list effects;

    BoundReactionWheel(hardware::ReactionWheel value, py::list effect_values)
        : native(std::move(value)), effects(std::move(effect_values)) {}
};


py::list fresh_list(const py::object& value) {
    if (value.is_none()) return py::list();
    if (PyList_Check(value.ptr()) != 0) return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

bool same(const py::object& left, const py::object& right) {
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}


template <std::size_t N>
std::array<double,N> read_fixed(py::handle value, const char* name) {
    auto array=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(value);
    if(!array||array.ndim()!=1||array.shape(0)!=static_cast<py::ssize_t>(N))
        throw py::value_error(std::string(name)+" has an invalid shape");
    std::array<double,N> result{}; auto input=array.unchecked<1>();
    for(std::size_t i=0;i<N;++i) result[i]=input(static_cast<py::ssize_t>(i));
    return result;
}
}

void bind_reaction_wheel(py::module_& module) {
    py::class_<BoundReactionWheel>(module, "ReactionWheel")
        .def(py::init([](py::handle position, py::handle orientation,
                         double rotational_inertia, double max_torque,
                         double max_momentum, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            auto native = hardware::ReactionWheel(read_fixed<3>(position, "position"),
                read_fixed<3>(orientation, "orientation"), rotational_inertia,
                max_torque, max_momentum, read_fixed<3>(misalignment, "misalignment"));
            return BoundReactionWheel(std::move(native), fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"),
            py::arg("rotational_inertia"), py::arg("max_torque"),
            py::arg("max_momentum"), py::arg("misalignment") = py::none(),
            py::arg("effects") = py::none())
        .def_property("position", [](const BoundReactionWheel& self) {
            return py::make_tuple(self.native.position[0], self.native.position[1], self.native.position[2]);
        }, [](BoundReactionWheel& self, py::handle value) { self.native.position = read_fixed<3>(value, "position"); })
        .def_property("orientation", [](const BoundReactionWheel& self) {
            return py::make_tuple(self.native.orientation[0], self.native.orientation[1], self.native.orientation[2]);
        }, [](BoundReactionWheel& self, py::handle value) { self.native.orientation = read_fixed<3>(value, "orientation"); })
        .def_property("rotational_inertia", [](const BoundReactionWheel& self) { return self.native.rotational_inertia; }, [](BoundReactionWheel& self, double value) { self.native.rotational_inertia = value; })
        .def_property("max_torque", [](const BoundReactionWheel& self) { return self.native.max_torque; }, [](BoundReactionWheel& self, double value) { self.native.max_torque = value; })
        .def_property("max_momentum", [](const BoundReactionWheel& self) { return self.native.max_momentum; }, [](BoundReactionWheel& self, double value) { self.native.max_momentum = value; })
        .def_property("misalignment", [](const BoundReactionWheel& self) {
            return py::make_tuple(self.native.misalignment[0], self.native.misalignment[1], self.native.misalignment[2]);
        }, [](BoundReactionWheel& self, py::handle value) { self.native.misalignment = read_fixed<3>(value, "misalignment"); })
        .def_readwrite("effects", &BoundReactionWheel::effects)
        .def_property_readonly("noises", [](const BoundReactionWheel& self) {
            return hardware::filter_effects(self.effects, "GenADCS.backends.python.noise", "Noise");
        })
        .def_property_readonly("biases", [](const BoundReactionWheel& self) {
            return hardware::filter_effects(self.effects, "GenADCS.backends.python.bias", "Bias");
        })
        .def("apply_errors", [](const BoundReactionWheel& self, py::object value,
                                  double dt, py::object rng) {
            return hardware::apply_effects(std::move(value), self.effects, dt, std::move(rng));
        }, py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const BoundReactionWheel& left,
                           const BoundReactionWheel& right) {
            return left.native.position == right.native.position
                && left.native.orientation == right.native.orientation
                && (left.native.rotational_inertia == right.native.rotational_inertia)
                && (left.native.max_torque == right.native.max_torque)
                && (left.native.max_momentum == right.native.max_momentum)
                && left.native.misalignment == right.native.misalignment
                && same(left.effects, right.effects);
        });

}

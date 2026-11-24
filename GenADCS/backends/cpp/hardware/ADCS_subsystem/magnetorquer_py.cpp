#include "adcs_subsystem.hpp"
#include "effects_py.hpp"
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
struct BoundMagnetorquer {
    hardware::Magnetorquer native;
    py::list effects;

    BoundMagnetorquer(hardware::Magnetorquer value, py::list effect_values)
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

void bind_magnetorquer(py::module_& module) {
    py::class_<BoundMagnetorquer>(module, "Magnetorquer")
        .def(py::init([](py::object position, py::object orientation,
                         double max_dipole, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            return BoundMagnetorquer(hardware::Magnetorquer(
                read_fixed<3>(position, "position"),
                read_fixed<3>(orientation, "orientation"), max_dipole,
                read_fixed<3>(misalignment, "misalignment")), fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"), py::arg("max_dipole"),
            py::arg("misalignment") = py::none(), py::arg("effects") = py::none())
        .def_property("position", [](const BoundMagnetorquer& self) {
            return py::make_tuple(self.native.position[0], self.native.position[1], self.native.position[2]);
        }, [](BoundMagnetorquer& self, py::handle value) {
            self.native.position = read_fixed<3>(value, "position");
        })
        .def_property("orientation", [](const BoundMagnetorquer& self) {
            return py::make_tuple(self.native.orientation[0], self.native.orientation[1], self.native.orientation[2]);
        }, [](BoundMagnetorquer& self, py::handle value) {
            self.native.orientation = read_fixed<3>(value, "orientation");
        })
        .def_property("max_dipole", [](const BoundMagnetorquer& self) { return self.native.max_dipole; }, [](BoundMagnetorquer& self, double value) { self.native.max_dipole = value; })
        .def_property("misalignment", [](const BoundMagnetorquer& self) {
            return py::make_tuple(self.native.misalignment[0], self.native.misalignment[1], self.native.misalignment[2]);
        }, [](BoundMagnetorquer& self, py::handle value) {
            self.native.misalignment = read_fixed<3>(value, "misalignment");
        })
        .def_readwrite("effects", &BoundMagnetorquer::effects)
        .def_property_readonly("noises", [](const BoundMagnetorquer& self) {
            return hardware::filter_effects(self.effects, "GenADCS.backends.python.noise", "Noise");
        })
        .def_property_readonly("biases", [](const BoundMagnetorquer& self) {
            return hardware::filter_effects(self.effects, "GenADCS.backends.python.bias", "Bias");
        })
        .def("apply_errors", [](const BoundMagnetorquer& self, py::object value,
                                 double dt, py::object rng) {
            return hardware::apply_effects(std::move(value), self.effects, dt, std::move(rng));
        }, py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const BoundMagnetorquer& left,
                           const BoundMagnetorquer& right) {
            return left.native.position == right.native.position
                && left.native.orientation == right.native.orientation
                && (left.native.max_dipole == right.native.max_dipole)
                && left.native.misalignment == right.native.misalignment
                && same(left.effects, right.effects);
        });

}

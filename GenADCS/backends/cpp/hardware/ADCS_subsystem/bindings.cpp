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
#include "../../helpers/math/rotations/sensor_kernels_pybind.hpp"

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

template <typename Values>
py::array_t<double> vector_array(const Values& values) {
    py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
    auto out = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i)
        out(i) = values[static_cast<std::size_t>(i)];
    return result;
}
template <typename Matrix>
py::array_t<double> matrix_array(const Matrix& values) {
    py::array_t<double> result({3, 3});
    auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j)
            out(i, j) = values[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return result;
}
template <typename Matrix>
py::array_t<double> matrix_array_2x2(const Matrix& values) {
    py::array_t<double> result({2, 2}); auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i=0;i<2;++i) for (py::ssize_t j=0;j<2;++j) out(i,j)=values[i][j];
    return result;
}
template <typename Matrix>
py::array_t<double> matrix_array_2x3(const Matrix& values) {
    py::array_t<double> result({2, 3}); auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i=0;i<2;++i) for (py::ssize_t j=0;j<3;++j) out(i,j)=values[i][j];
    return result;
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
template <typename Values>
py::array_t<double> vector_array_2(const Values& values) { return vector_array(values); }

}

PYBIND11_MODULE(_adcs_subsystem, module) {
    module.doc() = "C++ ADCS hardware subsystem containers";

    module.def("apply_effects", &hardware::apply_effects,
               py::arg("value"), py::arg("effects"), py::arg("dt"),
               py::arg("rng") = py::none());

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
        .def_readwrite("rotational_inertia", &BoundReactionWheel::native.rotational_inertia)
        .def_readwrite("max_torque", &BoundReactionWheel::native.max_torque)
        .def_readwrite("max_momentum", &BoundReactionWheel::native.max_momentum)
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
        .def_readwrite("max_dipole", &BoundMagnetorquer::native.max_dipole)
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

    py::class_<native_sensors::Gyroscope>(module, "Gyroscope")
        .def(py::init([](py::object position, py::object orientation,
                         py::object misalignment, double saturation_value,
                         py::object noise, py::object bias) {
            return native_sensors::Gyroscope(
                native_sensors::kernels::read_vector3(position, "position"),
                native_sensors::kernels::read_quaternion(orientation, "orientation"),
                native_sensors::kernels::read_quaternion(misalignment, "misalignment"),
                saturation_value,
                native_sensors::kernels::read_matrix(noise, "noise"),
                native_sensors::kernels::read_vector3(bias, "bias"));
        }),
             py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
             py::arg("saturation_value"), py::arg("noise"), py::arg("bias"))
        .def_property_readonly("position", [](const native_sensors::Gyroscope& s) { return vector_array(s.position); })
        .def_property_readonly("orientation", [](const native_sensors::Gyroscope& s) { return vector_array(s.orientation); })
        .def_property_readonly("misalignment", [](const native_sensors::Gyroscope& s) { return vector_array(s.misalignment); })
        .def_readonly("saturation_value", &native_sensors::Gyroscope::saturation_value)
        .def_property_readonly("noise", [](const native_sensors::Gyroscope& s) { return matrix_array(s.noise); })
        .def_property_readonly("noise_cholesky", [](const native_sensors::Gyroscope& s) { return matrix_array(s.noise_cholesky); })
        .def_property_readonly("bias", [](const native_sensors::Gyroscope& s) { return vector_array(s.bias); })
        .def_readwrite("enabled", &native_sensors::Gyroscope::enabled)
        .def_readwrite("seed", &native_sensors::Gyroscope::seed)
        .def("noiseless_biasfree_measurement", [](const native_sensors::Gyroscope& s, py::object rate) {
            return native_sensors::kernels::array_vector(s.noiseless_biasfree_measurement(native_sensors::kernels::read_vector3(rate, "angular_velocity_body")));
        })
        .def("noiseless_biased_measurement", [](const native_sensors::Gyroscope& s, py::object rate) {
            return native_sensors::kernels::array_vector(s.noiseless_biased_measurement(native_sensors::kernels::read_vector3(rate, "angular_velocity_body")));
        })
        .def("noisy_biasfree_measurement", [](native_sensors::Gyroscope& s, py::object rate) {
            return native_sensors::kernels::array_vector(s.noisy_biasfree_measurement(native_sensors::kernels::read_vector3(rate, "angular_velocity_body")));
        })
        .def("noisy_biased_measurement", [](native_sensors::Gyroscope& s, py::object rate) {
            return native_sensors::kernels::array_vector(s.noisy_biased_measurement(native_sensors::kernels::read_vector3(rate, "angular_velocity_body")));
        })
        .def("jacobian_angular_velocity", [](const native_sensors::Gyroscope& s) { return matrix_array(s.jacobian_angular_velocity()); })
        .def("jacobian_bias", [](const native_sensors::Gyroscope& s) { return matrix_array(s.jacobian_bias()); })
        .def("availability", &native_sensors::Gyroscope::availability);

    py::class_<native_sensors::Magnetometer>(module, "Magnetometer")
        .def(py::init([](py::object position, py::object orientation,
                         py::object misalignment, double saturation_value,
                         py::object noise, py::object bias) {
            return native_sensors::Magnetometer(native_sensors::kernels::read_vector3(position, "position"),
                native_sensors::kernels::read_quaternion(orientation, "orientation"),
                native_sensors::kernels::read_quaternion(misalignment, "misalignment"), saturation_value,
                native_sensors::kernels::read_matrix(noise, "noise"),
                native_sensors::kernels::read_vector3(bias, "bias"));
        }), py::arg("position"), py::arg("orientation"), py::arg("misalignment"),
             py::arg("saturation_value"), py::arg("noise"), py::arg("bias"))
        .def_property_readonly("position", [](const native_sensors::Magnetometer& s) { return vector_array(s.position); })
        .def_property_readonly("orientation", [](const native_sensors::Magnetometer& s) { return vector_array(s.orientation); })
        .def_property_readonly("misalignment", [](const native_sensors::Magnetometer& s) { return vector_array(s.misalignment); })
        .def_readonly("saturation_value", &native_sensors::Magnetometer::saturation_value)
        .def_property_readonly("noise", [](const native_sensors::Magnetometer& s) { return matrix_array(s.noise); })
        .def_property_readonly("noise_cholesky", [](const native_sensors::Magnetometer& s) { return matrix_array(s.noise_cholesky); })
        .def_property_readonly("bias", [](const native_sensors::Magnetometer& s) { return vector_array(s.bias); })
        .def_readwrite("enabled", &native_sensors::Magnetometer::enabled)
        .def_readwrite("seed", &native_sensors::Magnetometer::seed)
        .def("noiseless_biasfree_measurement", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q) { return vector_array(s.noiseless_biasfree_measurement(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"))); })
        .def("noiseless_biased_measurement", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q) { return vector_array(s.noiseless_biased_measurement(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"))); })
        .def("noisy_biasfree_measurement", [](native_sensors::Magnetometer& s, py::handle f, py::handle q) { return vector_array(s.noisy_biasfree_measurement(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"))); })
        .def("noisy_biased_measurement", [](native_sensors::Magnetometer& s, py::handle f, py::handle q) { return vector_array(s.noisy_biased_measurement(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"))); })
        .def("jacobian_bias", [](const native_sensors::Magnetometer& s) { return matrix_array(s.jacobian_bias()); })
        .def("jacobian_relative_quaternion", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q, py::handle d) { return matrix_array(s.jacobian_relative_quaternion(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"), native_sensors::kernels::read_quaternion(d, "delta_quaternion"))); })
        .def("jacobian_relative_mrp", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q, py::handle d) { return matrix_array(s.jacobian_relative_mrp(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"), native_sensors::kernels::read_vector3(d, "delta_mrp"))); })
        .def("jacobian_relative_angle_vector", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q, py::handle d) { return matrix_array(s.jacobian_relative_angle_vector(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"), native_sensors::kernels::read_vector3(d, "delta_angle"))); })
        .def("availability", &native_sensors::Magnetometer::availability)
        .def("fisher_information_matrix", [](const native_sensors::Magnetometer& s, py::handle f, py::handle q, py::handle c) { return matrix_array(s.fisher_information_matrix(native_sensors::kernels::read_vector3(f, "field_inertial"), native_sensors::kernels::read_quaternion(q, "attitude"), native_sensors::kernels::read_matrix(c, "field_covariance"))); });

    py::class_<native_sensors::Accelerometer>(module, "Accelerometer")
        .def(py::init([](py::object position,py::object orientation,py::object misalignment,
                         double saturation,py::object noise,py::object bias) {
            return native_sensors::Accelerometer(
                native_sensors::kernels::read_vector3(position, "position"),
                native_sensors::kernels::read_quaternion(orientation, "orientation"),
                native_sensors::kernels::read_quaternion(misalignment, "misalignment"),
                saturation, native_sensors::kernels::read_matrix(noise, "noise"),
                native_sensors::kernels::read_vector3(bias, "bias"));
        }),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),
             py::arg("saturation_value"),py::arg("noise"),py::arg("bias"))
        .def_property_readonly("position",[](const native_sensors::Accelerometer& s){return vector_array(s.position);})
        .def_property_readonly("orientation",[](const native_sensors::Accelerometer& s){return vector_array(s.orientation);})
        .def_property_readonly("misalignment",[](const native_sensors::Accelerometer& s){return vector_array(s.misalignment);})
        .def_readonly("saturation_value",&native_sensors::Accelerometer::saturation_value)
        .def_property_readonly("noise",[](const native_sensors::Accelerometer& s){return matrix_array(s.noise);})
        .def_property_readonly("noise_cholesky",[](const native_sensors::Accelerometer& s){return matrix_array(s.noise_cholesky);})
        .def_property_readonly("bias",[](const native_sensors::Accelerometer& s){return vector_array(s.bias);})
        .def_readwrite("enabled",&native_sensors::Accelerometer::enabled)
        .def_readwrite("seed",&native_sensors::Accelerometer::seed)
        .def("noiseless_biasfree_measurement",[](const native_sensors::Accelerometer& s, py::object a, py::object g, py::object q) {
            return vector_array(s.noiseless_biasfree_measurement(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude")));
        })
        .def("noiseless_biased_measurement",[](const native_sensors::Accelerometer& s, py::object a, py::object g, py::object q) {
            return vector_array(s.noiseless_biased_measurement(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude")));
        })
        .def("noisy_biasfree_measurement",[](native_sensors::Accelerometer& s, py::object a, py::object g, py::object q) {
            return vector_array(s.noisy_biasfree_measurement(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude")));
        })
        .def("noisy_biased_measurement",[](native_sensors::Accelerometer& s, py::object a, py::object g, py::object q) {
            return vector_array(s.noisy_biased_measurement(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude")));
        })
        .def("jacobian_relative_quaternion",[](const native_sensors::Accelerometer& s, py::object a, py::object g, py::object q, py::object d) {
            return matrix_array(s.jacobian_relative_quaternion(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude"), native_sensors::kernels::read_quaternion(d,"delta_quaternion")));
        })
        .def("jacobian_relative_mrp",[](const native_sensors::Accelerometer& s, py::object a, py::object g, py::object q, py::object d) {
            return matrix_array(s.jacobian_relative_mrp(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude"), native_sensors::kernels::read_vector3(d,"delta_mrp")));
        })
        .def("jacobian_relative_angle_vector",[](const native_sensors::Accelerometer& s, py::object a, py::object g, py::object q, py::object d) {
            return matrix_array(s.jacobian_relative_angle_vector(native_sensors::kernels::read_vector3(a,"acceleration_inertial"), native_sensors::kernels::read_vector3(g,"gravity_inertial"), native_sensors::kernels::read_quaternion(q,"attitude"), native_sensors::kernels::read_vector3(d,"delta_angle")));
        })
        .def("jacobian_bias",[](const native_sensors::Accelerometer& s) { return matrix_array(s.jacobian_bias()); })
        .def("availability",&native_sensors::Accelerometer::availability);

    py::class_<native_sensors::CoarseSunSensor>(module,"CoarseSunSensor")
        .def(py::init([](py::object p,py::object q,py::object m,py::object n,double sensitivity,
                         double saturation,double variance,double bias,double fov){
            return native_sensors::CoarseSunSensor(
                native_sensors::kernels::read_vector3(p,"position"),
                native_sensors::kernels::read_quaternion(q,"orientation"),
                native_sensors::kernels::read_quaternion(m,"misalignment"),
                native_sensors::kernels::read_vector3(n,"normal"),sensitivity,saturation,variance,bias,fov);
        }),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),py::arg("normal"),
             py::arg("sensitivity"),py::arg("saturation_value"),py::arg("noise_variance"),py::arg("bias"),py::arg("fov_half_angle"))
        .def_property_readonly("position",[](const native_sensors::CoarseSunSensor&s){return vector_array(s.position);})
        .def_property_readonly("orientation",[](const native_sensors::CoarseSunSensor&s){return vector_array(s.orientation);})
        .def_property_readonly("misalignment",[](const native_sensors::CoarseSunSensor&s){return vector_array(s.misalignment);})
        .def_property_readonly("normal",[](const native_sensors::CoarseSunSensor&s){return vector_array(s.normal);})
        .def_readonly("sensitivity",&native_sensors::CoarseSunSensor::sensitivity)
        .def_readonly("saturation_value",&native_sensors::CoarseSunSensor::saturation_value)
        .def_readonly("noise_variance",&native_sensors::CoarseSunSensor::noise_variance)
        .def_readonly("noise_std",&native_sensors::CoarseSunSensor::noise_std)
        .def_readonly("bias",&native_sensors::CoarseSunSensor::bias)
        .def_readonly("fov_half_angle",&native_sensors::CoarseSunSensor::fov_half_angle)
        .def_readwrite("enabled",&native_sensors::CoarseSunSensor::enabled)
        .def_readwrite("seed",&native_sensors::CoarseSunSensor::seed)
        .def("noiseless_biasfree_measurement",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q){return s.noiseless_biasfree_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"));})
        .def("noiseless_biased_measurement",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q){return s.noiseless_biased_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"));})
        .def("noisy_biasfree_measurement",[](native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q){return s.noisy_biasfree_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"));})
        .def("noisy_biased_measurement",[](native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q){return s.noisy_biased_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"));})
        .def("jacobian_bias",&native_sensors::CoarseSunSensor::jacobian_bias)
        .def("jacobian_relative_quaternion",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q,py::handle d){return vector_array(s.jacobian_relative_quaternion(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_quaternion(d,"delta_quaternion")));})
        .def("jacobian_relative_mrp",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q,py::handle d){return vector_array(s.jacobian_relative_mrp(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_vector3(d,"delta_mrp")));})
        .def("jacobian_relative_angle_vector",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q,py::handle d){return vector_array(s.jacobian_relative_angle_vector(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_vector3(d,"delta_angle")));})
        .def("availability",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q,bool visible){return s.availability(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),visible);},py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true)
        .def("fisher_information_matrix",[](const native_sensors::CoarseSunSensor& s,py::handle sun,py::handle q,bool visible){return matrix_array(s.fisher_information_matrix(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),visible));},py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true);

    py::class_<native_sensors::FineSunSensor>(module,"FineSunSensor")
        .def(py::init([](py::handle p,py::handle q,py::handle m,py::handle n,py::handle b,py::handle f){
            native_sensors::FineSunSensor::M2 covariance{};
            auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(n);
            if(!a||a.ndim()!=2||a.shape(0)!=2||a.shape(1)!=2)throw py::value_error("noise must be 2x2");
            auto values=a.unchecked<2>(); for(int i=0;i<2;++i)for(int j=0;j<2;++j)covariance[i][j]=values(i,j);
            const auto bias=read_fixed<2>(b,"bias");
            const auto fov=read_fixed<2>(f,"fov_half_angles");
            return native_sensors::FineSunSensor(native_sensors::kernels::read_vector3(p,"position"),
                native_sensors::kernels::normalize(native_sensors::kernels::read_quaternion(q,"orientation")),
                native_sensors::kernels::normalize(native_sensors::kernels::read_quaternion(m,"misalignment")),
                covariance,bias,fov);
        }),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),py::arg("noise"),py::arg("bias"),py::arg("fov_half_angles"))
        .def_property_readonly("position",[](const native_sensors::FineSunSensor&s){return vector_array(s.position);})
        .def_property_readonly("orientation",[](const native_sensors::FineSunSensor&s){return vector_array(s.orientation);})
        .def_property_readonly("misalignment",[](const native_sensors::FineSunSensor&s){return vector_array(s.misalignment);})
        .def_readwrite("enabled",&native_sensors::FineSunSensor::enabled).def_readwrite("seed",&native_sensors::FineSunSensor::seed)
        .def_readonly("noise",&native_sensors::FineSunSensor::noise).def_readonly("bias",&native_sensors::FineSunSensor::bias)
        .def_readonly("fov_half_angles",&native_sensors::FineSunSensor::fov_half_angles)
        .def("noiseless_biasfree_measurement",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q){return vector_array_2(s.noiseless_biasfree_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude")));})
        .def("noiseless_biased_measurement",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q){return vector_array_2(s.noiseless_biased_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude")));})
        .def("noisy_biasfree_measurement",[](native_sensors::FineSunSensor&s,py::handle sun,py::handle q){return vector_array_2(s.noisy_biasfree_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude")));})
        .def("noisy_biased_measurement",[](native_sensors::FineSunSensor&s,py::handle sun,py::handle q){return vector_array_2(s.noisy_biased_measurement(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude")));})
        .def("jacobian_bias",[](const native_sensors::FineSunSensor&s){return matrix_array_2x2(s.jacobian_bias());})
        .def("jacobian_relative_quaternion",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q,py::handle d){return matrix_array_2x3(s.jacobian_relative_quaternion(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_quaternion(d,"delta_quaternion")));})
        .def("jacobian_relative_mrp",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q,py::handle d){return matrix_array_2x3(s.jacobian_relative_mrp(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_vector3(d,"delta_mrp")));})
        .def("jacobian_relative_angle_vector",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q,py::handle d){return matrix_array_2x3(s.jacobian_relative_angle_vector(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),native_sensors::kernels::read_vector3(d,"delta_angle")));})
        .def("availability",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q,bool visible){return s.availability(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),visible);},py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true)
        .def("fisher_information_matrix",[](const native_sensors::FineSunSensor&s,py::handle sun,py::handle q,bool visible){return matrix_array(s.fisher_information_matrix(native_sensors::kernels::read_vector3(sun,"sun_inertial"),native_sensors::kernels::read_quaternion(q,"attitude"),visible));},py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true);

    py::class_<native_sensors::FineEarthSensor,native_sensors::FineSunSensor>(module,"FineEarthSensor")
        .def(py::init([](py::object p,py::object q,py::object m,py::object n,py::object b,py::object f){
            native_sensors::FineSunSensor::M2 covariance{};
            auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(n);
            if(!a||a.ndim()!=2||a.shape(0)!=2||a.shape(1)!=2)throw py::value_error("noise must be 2x2");
            auto values=a.unchecked<2>(); for(int i=0;i<2;++i)for(int j=0;j<2;++j)covariance[i][j]=values(i,j);
            return native_sensors::FineEarthSensor(native_sensors::kernels::read_vector3(p,"position"),
                native_sensors::kernels::normalize(native_sensors::kernels::read_quaternion(q,"orientation")),
                native_sensors::kernels::normalize(native_sensors::kernels::read_quaternion(m,"misalignment")),
                covariance,read_fixed<2>(b,"bias"),read_fixed<2>(f,"fov_half_angles"));
        }),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),py::arg("noise"),py::arg("bias"),py::arg("fov_half_angles"))
        .def("availability",[](const native_sensors::FineEarthSensor& s,py::handle earth,py::handle attitude,bool visible){return s.availability(native_sensors::kernels::read_vector3(earth,"earth_inertial"),native_sensors::kernels::read_quaternion(attitude,"attitude"),visible);},
             py::arg("earth_inertial"),py::arg("attitude"),py::arg("earth_visible")=true)
        .def("fisher_information_matrix",[](const native_sensors::FineEarthSensor& s,py::handle earth,py::handle attitude,bool visible){return matrix_array(s.fisher_information_matrix(native_sensors::kernels::read_vector3(earth,"earth_inertial"),native_sensors::kernels::read_quaternion(attitude,"attitude"),visible));},
             py::arg("earth_inertial"),py::arg("attitude"),py::arg("earth_visible")=true);
}

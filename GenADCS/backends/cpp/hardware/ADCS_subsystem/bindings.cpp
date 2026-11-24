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

template <typename Values>
py::array_t<double> vector_array(const Values& values) {
    py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
    auto out = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i)
        out(i) = values[static_cast<std::size_t>(i)];
    return result;
}
py::array_t<double> matrix_array(const native_sensors::Matrix3& values) {
    py::array_t<double> result({3, 3});
    auto out = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i)
        for (py::ssize_t j = 0; j < 3; ++j)
            out(i, j) = values[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return result;
}
}

PYBIND11_MODULE(_adcs_subsystem, module) {
    module.doc() = "C++ ADCS hardware subsystem containers";

    module.def("apply_effects", &hardware::apply_effects,
               py::arg("value"), py::arg("effects"), py::arg("dt"),
               py::arg("rng") = py::none());

    py::class_<hardware::ReactionWheel>(module, "ReactionWheel")
        .def(py::init([](py::object position, py::object orientation,
                         double rotational_inertia, double max_torque,
                         double max_momentum, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            return hardware::ReactionWheel(std::move(position), std::move(orientation),
                rotational_inertia, max_torque, max_momentum, std::move(misalignment),
                fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"),
            py::arg("rotational_inertia"), py::arg("max_torque"),
            py::arg("max_momentum"), py::arg("misalignment") = py::none(),
            py::arg("effects") = py::none())
        .def_readwrite("position", &hardware::ReactionWheel::position)
        .def_readwrite("orientation", &hardware::ReactionWheel::orientation)
        .def_readwrite("rotational_inertia", &hardware::ReactionWheel::rotational_inertia)
        .def_readwrite("max_torque", &hardware::ReactionWheel::max_torque)
        .def_readwrite("max_momentum", &hardware::ReactionWheel::max_momentum)
        .def_readwrite("misalignment", &hardware::ReactionWheel::misalignment)
        .def_readwrite("effects", &hardware::ReactionWheel::effects)
        .def_property_readonly("noises", &hardware::ReactionWheel::noises)
        .def_property_readonly("biases", &hardware::ReactionWheel::biases)
        .def("apply_errors", &hardware::ReactionWheel::apply_errors,
             py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const hardware::ReactionWheel& left,
                           const hardware::ReactionWheel& right) {
            return same(left.position, right.position) && same(left.orientation, right.orientation)
                && (left.rotational_inertia == right.rotational_inertia)
                && (left.max_torque == right.max_torque)
                && (left.max_momentum == right.max_momentum)
                && same(left.misalignment, right.misalignment) && same(left.effects, right.effects);
        });

    py::class_<hardware::Magnetorquer>(module, "Magnetorquer")
        .def(py::init([](py::object position, py::object orientation,
                         double max_dipole, py::object misalignment,
                         py::object effects) {
            if (misalignment.is_none()) misalignment = py::make_tuple(0.0, 0.0, 0.0);
            return hardware::Magnetorquer(std::move(position), std::move(orientation),
                max_dipole, std::move(misalignment), fresh_list(effects));
        }), py::arg("position"), py::arg("orientation"), py::arg("max_dipole"),
            py::arg("misalignment") = py::none(), py::arg("effects") = py::none())
        .def_readwrite("position", &hardware::Magnetorquer::position)
        .def_readwrite("orientation", &hardware::Magnetorquer::orientation)
        .def_readwrite("max_dipole", &hardware::Magnetorquer::max_dipole)
        .def_readwrite("misalignment", &hardware::Magnetorquer::misalignment)
        .def_readwrite("effects", &hardware::Magnetorquer::effects)
        .def_property_readonly("noises", &hardware::Magnetorquer::noises)
        .def_property_readonly("biases", &hardware::Magnetorquer::biases)
        .def("apply_errors", &hardware::Magnetorquer::apply_errors,
             py::arg("value"), py::arg("dt"), py::arg("rng") = py::none())
        .def("__eq__", [](const hardware::Magnetorquer& left,
                           const hardware::Magnetorquer& right) {
            return same(left.position, right.position) && same(left.orientation, right.orientation)
                && (left.max_dipole == right.max_dipole)
                && same(left.misalignment, right.misalignment) && same(left.effects, right.effects);
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
            return native_sensors::Gyroscope(position, orientation, misalignment,
                                               saturation_value, noise, bias);
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
        .def("noiseless_biasfree_measurement", &native_sensors::Gyroscope::noiseless_biasfree_measurement)
        .def("noiseless_biased_measurement", &native_sensors::Gyroscope::noiseless_biased_measurement)
        .def("noisy_biasfree_measurement", &native_sensors::Gyroscope::noisy_biasfree_measurement)
        .def("noisy_biased_measurement", &native_sensors::Gyroscope::noisy_biased_measurement)
        .def("jacobian_angular_velocity", &native_sensors::Gyroscope::jacobian_angular_velocity)
        .def("jacobian_bias", &native_sensors::Gyroscope::jacobian_bias)
        .def("availability", &native_sensors::Gyroscope::availability);

    py::class_<native_sensors::Magnetometer>(module, "Magnetometer")
        .def(py::init([](py::object position, py::object orientation,
                         py::object misalignment, double saturation_value,
                         py::object noise, py::object bias) {
            return native_sensors::Magnetometer(position, orientation, misalignment,
                                                  saturation_value, noise, bias);
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
        .def("noiseless_biasfree_measurement", &native_sensors::Magnetometer::noiseless_biasfree_measurement)
        .def("noiseless_biased_measurement", &native_sensors::Magnetometer::noiseless_biased_measurement)
        .def("noisy_biasfree_measurement", &native_sensors::Magnetometer::noisy_biasfree_measurement)
        .def("noisy_biased_measurement", &native_sensors::Magnetometer::noisy_biased_measurement)
        .def("jacobian_bias", &native_sensors::Magnetometer::jacobian_bias)
        .def("jacobian_relative_quaternion", &native_sensors::Magnetometer::jacobian_relative_quaternion)
        .def("jacobian_relative_mrp", &native_sensors::Magnetometer::jacobian_relative_mrp)
        .def("jacobian_relative_angle_vector", &native_sensors::Magnetometer::jacobian_relative_angle_vector)
        .def("availability", &native_sensors::Magnetometer::availability)
        .def("fisher_information_matrix", &native_sensors::Magnetometer::fisher_information_matrix);

    py::class_<native_sensors::Accelerometer>(module, "Accelerometer")
        .def(py::init([](py::object position,py::object orientation,py::object misalignment,
                         double saturation,py::object noise,py::object bias) {
            return native_sensors::Accelerometer(position,orientation,misalignment,saturation,noise,bias);
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
        .def("noiseless_biasfree_measurement",&native_sensors::Accelerometer::noiseless_biasfree_measurement)
        .def("noiseless_biased_measurement",&native_sensors::Accelerometer::noiseless_biased_measurement)
        .def("noisy_biasfree_measurement",&native_sensors::Accelerometer::noisy_biasfree_measurement)
        .def("noisy_biased_measurement",&native_sensors::Accelerometer::noisy_biased_measurement)
        .def("jacobian_relative_quaternion",&native_sensors::Accelerometer::jacobian_relative_quaternion)
        .def("jacobian_relative_mrp",&native_sensors::Accelerometer::jacobian_relative_mrp)
        .def("jacobian_relative_angle_vector",&native_sensors::Accelerometer::jacobian_relative_angle_vector)
        .def("jacobian_bias",&native_sensors::Accelerometer::jacobian_bias)
        .def("availability",&native_sensors::Accelerometer::availability);

    py::class_<native_sensors::CoarseSunSensor>(module,"CoarseSunSensor")
        .def(py::init([](py::object p,py::object q,py::object m,py::object n,double sensitivity,
                         double saturation,double variance,double bias,double fov){
            return native_sensors::CoarseSunSensor(p,q,m,n,sensitivity,saturation,variance,bias,fov);
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
        .def("noiseless_biasfree_measurement",&native_sensors::CoarseSunSensor::noiseless_biasfree_measurement)
        .def("noiseless_biased_measurement",&native_sensors::CoarseSunSensor::noiseless_biased_measurement)
        .def("noisy_biasfree_measurement",&native_sensors::CoarseSunSensor::noisy_biasfree_measurement)
        .def("noisy_biased_measurement",&native_sensors::CoarseSunSensor::noisy_biased_measurement)
        .def("jacobian_bias",&native_sensors::CoarseSunSensor::jacobian_bias)
        .def("jacobian_relative_quaternion",&native_sensors::CoarseSunSensor::jacobian_relative_quaternion)
        .def("jacobian_relative_mrp",&native_sensors::CoarseSunSensor::jacobian_relative_mrp)
        .def("jacobian_relative_angle_vector",&native_sensors::CoarseSunSensor::jacobian_relative_angle_vector)
        .def("availability",&native_sensors::CoarseSunSensor::availability,py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true)
        .def("fisher_information_matrix",&native_sensors::CoarseSunSensor::fisher_information_matrix,py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true);

    py::class_<native_sensors::FineSunSensor>(module,"FineSunSensor")
        .def(py::init([](py::object p,py::object q,py::object m,py::object n,py::object b,py::object f){return native_sensors::FineSunSensor(p,q,m,n,b,f);}),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),py::arg("noise"),py::arg("bias"),py::arg("fov_half_angles"))
        .def_property_readonly("position",[](const native_sensors::FineSunSensor&s){return vector_array(s.position);})
        .def_property_readonly("orientation",[](const native_sensors::FineSunSensor&s){return vector_array(s.orientation);})
        .def_property_readonly("misalignment",[](const native_sensors::FineSunSensor&s){return vector_array(s.misalignment);})
        .def_readwrite("enabled",&native_sensors::FineSunSensor::enabled).def_readwrite("seed",&native_sensors::FineSunSensor::seed)
        .def("noiseless_biasfree_measurement",&native_sensors::FineSunSensor::noiseless_biasfree_measurement).def("noiseless_biased_measurement",&native_sensors::FineSunSensor::noiseless_biased_measurement)
        .def("noisy_biasfree_measurement",&native_sensors::FineSunSensor::noisy_biasfree_measurement).def("noisy_biased_measurement",&native_sensors::FineSunSensor::noisy_biased_measurement)
        .def("jacobian_bias",&native_sensors::FineSunSensor::jacobian_bias).def("jacobian_relative_quaternion",&native_sensors::FineSunSensor::jacobian_relative_quaternion)
        .def("jacobian_relative_mrp",&native_sensors::FineSunSensor::jacobian_relative_mrp).def("jacobian_relative_angle_vector",&native_sensors::FineSunSensor::jacobian_relative_angle_vector)
        .def("availability",&native_sensors::FineSunSensor::availability,py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true)
        .def("fisher_information_matrix",&native_sensors::FineSunSensor::fisher_information_matrix,py::arg("sun_inertial"),py::arg("attitude"),py::arg("sun_visible")=true);

    py::class_<native_sensors::FineEarthSensor,native_sensors::FineSunSensor>(module,"FineEarthSensor")
        .def(py::init([](py::object p,py::object q,py::object m,py::object n,py::object b,py::object f){
            return native_sensors::FineEarthSensor(p,q,m,n,b,f);
        }),py::arg("position"),py::arg("orientation"),py::arg("misalignment"),py::arg("noise"),py::arg("bias"),py::arg("fov_half_angles"))
        .def("availability",[](const native_sensors::FineEarthSensor& s,py::handle earth,py::handle attitude,bool visible){return s.availability(earth,attitude,visible);},
             py::arg("earth_inertial"),py::arg("attitude"),py::arg("earth_visible")=true)
        .def("fisher_information_matrix",[](const native_sensors::FineEarthSensor& s,py::handle earth,py::handle attitude,bool visible){return s.fisher_information_matrix(earth,attitude,visible);},
             py::arg("earth_inertial"),py::arg("attitude"),py::arg("earth_visible")=true);
}

#include "magnetometer.hpp"
#include "philox.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace omni::hardware::sensors {

Magnetometer::Magnetometer(py::handle p, py::handle q, py::handle mis, double saturation,
                           py::handle covariance, py::handle b)
    : position(kernels::read_vector3(p,"position")), orientation(kernels::normalize(kernels::read_quaternion(q,"orientation"))),
      misalignment(kernels::normalize(kernels::read_quaternion(mis,"misalignment"))), saturation_value(saturation),
      noise(kernels::read_matrix(covariance,"noise")), noise_cholesky(kernels::cholesky(noise)), bias(kernels::read_vector3(b,"bias")) {
    if (!(saturation_value > 0.0)) throw py::value_error("saturation_value must be positive");
}

MagnetometerVector3 Magnetometer::sensor_field(py::handle field, py::handle attitude) const {
    return kernels::rotate_inertial_to_sensor(kernels::read_vector3(field,"field_inertial"),
        kernels::read_quaternion(attitude,"attitude"),orientation,misalignment);
}
py::array_t<double> Magnetometer::measure(py::handle field, py::handle attitude, bool with_bias, bool with_noise) {
    MagnetometerVector3 draw{};
    if (with_noise) {
        const auto values = omni::helpers::random::standard_normals(seed, random_tick++, 0x40000000U, 3);
        for (std::size_t i = 0; i < 3; ++i) draw[i] = values[i];
    }
    return kernels::array_vector(kernels::vector_measurement(
        kernels::read_vector3(field, "field_inertial"),
        kernels::read_quaternion(attitude, "attitude"), orientation, misalignment,
        bias, noise_cholesky, saturation_value, with_bias, with_noise,
        with_noise ? &draw : nullptr));
}
py::array_t<double> Magnetometer::noiseless_biasfree_measurement(py::handle f,py::handle q) const { return const_cast<Magnetometer*>(this)->measure(f,q,false,false); }
py::array_t<double> Magnetometer::noiseless_biased_measurement(py::handle f,py::handle q) const { return const_cast<Magnetometer*>(this)->measure(f,q,true,false); }
py::array_t<double> Magnetometer::noisy_biasfree_measurement(py::handle f,py::handle q) { return measure(f,q,false,true); }
py::array_t<double> Magnetometer::noisy_biased_measurement(py::handle f,py::handle q) { return measure(f,q,true,true); }
py::array_t<double> Magnetometer::jacobian_bias() const { return kernels::array_matrix(kernels::identity()); }

py::array_t<double> Magnetometer::jacobian_relative_quaternion(py::handle field,py::handle attitude,py::handle delta) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_quaternion(sensor_field(field,attitude),
        kernels::read_quaternion(delta,"delta_quaternion")));
}
py::array_t<double> Magnetometer::jacobian_relative_mrp(py::handle field,py::handle attitude,py::handle delta) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_mrp(sensor_field(field,attitude),
        kernels::read_vector3(delta,"delta_mrp")));
}
py::array_t<double> Magnetometer::jacobian_relative_angle_vector(py::handle field,py::handle attitude,py::handle delta) const {
    return kernels::array_matrix(kernels::attitude_vector_jacobian_angle(sensor_field(field,attitude),
        kernels::read_vector3(delta,"delta_angle")));
}
bool Magnetometer::availability() const { return enabled; }
py::array_t<double> Magnetometer::fisher_information_matrix(py::handle field,py::handle attitude,py::handle field_covariance) const {
    if(!enabled) return kernels::array_matrix(MagnetometerMatrix3{});
    return kernels::array_matrix(kernels::magnetometer_information(
        kernels::read_vector3(field,"field_inertial"),kernels::read_quaternion(attitude,"attitude"),
        orientation,misalignment,noise,kernels::read_matrix(field_covariance,"field_covariance")));
}
}

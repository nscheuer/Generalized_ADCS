#ifndef GENADCS_HARDWARE_SENSORS_MAGNETOMETER_HPP
#define GENADCS_HARDWARE_SENSORS_MAGNETOMETER_HPP

#include <pybind11/numpy.h>

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
using MagnetometerVector3 = std::array<double, 3>;
using MagnetometerQuaternion = std::array<double, 4>;
using MagnetometerMatrix3 = std::array<std::array<double, 3>, 3>;

struct Magnetometer {
    Magnetometer(pybind11::handle position, pybind11::handle orientation,
                 pybind11::handle misalignment, double saturation_value,
                 pybind11::handle noise, pybind11::handle bias);

    pybind11::array_t<double> noiseless_biasfree_measurement(pybind11::handle field, pybind11::handle attitude) const;
    pybind11::array_t<double> noiseless_biased_measurement(pybind11::handle field, pybind11::handle attitude) const;
    pybind11::array_t<double> noisy_biasfree_measurement(pybind11::handle field, pybind11::handle attitude);
    pybind11::array_t<double> noisy_biased_measurement(pybind11::handle field, pybind11::handle attitude);
    pybind11::array_t<double> jacobian_bias() const;
    pybind11::array_t<double> jacobian_relative_quaternion(pybind11::handle field, pybind11::handle attitude, pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_mrp(pybind11::handle field, pybind11::handle attitude, pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_angle_vector(pybind11::handle field, pybind11::handle attitude, pybind11::handle delta) const;
    bool availability() const;
    pybind11::array_t<double> fisher_information_matrix(pybind11::handle field, pybind11::handle attitude, pybind11::handle field_covariance) const;

    MagnetometerVector3 position{};
    MagnetometerQuaternion orientation{};
    MagnetometerQuaternion misalignment{};
    double saturation_value;
    MagnetometerMatrix3 noise{};
    MagnetometerMatrix3 noise_cholesky{};
    MagnetometerVector3 bias{};
    bool enabled = true;
    std::uint64_t seed = 0;
    std::uint64_t random_tick = 0;

private:
    pybind11::array_t<double> measure(pybind11::handle field, pybind11::handle attitude,
                                      bool with_bias, bool with_noise);
    MagnetometerVector3 sensor_field(pybind11::handle field, pybind11::handle attitude) const;
};
}

#endif

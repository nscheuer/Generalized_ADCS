#ifndef GENADCS_HARDWARE_SENSORS_GYROSCOPE_HPP
#define GENADCS_HARDWARE_SENSORS_GYROSCOPE_HPP

#include <pybind11/numpy.h>

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
using Vector3 = std::array<double, 3>;
using Quaternion = std::array<double, 4>;
using Matrix3 = std::array<std::array<double, 3>, 3>;

struct Gyroscope {
    Gyroscope(pybind11::handle position, pybind11::handle orientation,
              pybind11::handle misalignment, double saturation_value,
              pybind11::handle noise, pybind11::handle bias);

    pybind11::array_t<double> noiseless_biasfree_measurement(pybind11::handle rate) const;
    pybind11::array_t<double> noiseless_biased_measurement(pybind11::handle rate) const;
    pybind11::array_t<double> noisy_biasfree_measurement(pybind11::handle rate);
    pybind11::array_t<double> noisy_biased_measurement(pybind11::handle rate);
    pybind11::array_t<double> jacobian_angular_velocity() const;
    pybind11::array_t<double> jacobian_bias() const;
    bool availability() const;

    Vector3 position{};
    Quaternion orientation{};
    Quaternion misalignment{};
    double saturation_value;
    Matrix3 noise{};
    Matrix3 noise_cholesky{};
    Vector3 bias{};
    bool enabled = true;
    std::uint64_t seed = 0;
    std::uint64_t random_tick = 0;

private:
    pybind11::array_t<double> measure(pybind11::handle rate, bool with_bias, bool with_noise);
};
}

#endif

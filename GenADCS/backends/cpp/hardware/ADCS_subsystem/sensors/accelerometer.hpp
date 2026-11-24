#ifndef GENADCS_HARDWARE_SENSORS_ACCELEROMETER_HPP
#define GENADCS_HARDWARE_SENSORS_ACCELEROMETER_HPP
#include <pybind11/numpy.h>
#include <array>
#include <cstdint>
namespace omni::hardware::sensors {
struct Accelerometer {
    using V = std::array<double,3>; using Q = std::array<double,4>; using M = std::array<std::array<double,3>,3>;
    Accelerometer(pybind11::handle position, pybind11::handle orientation,
                   pybind11::handle misalignment, double saturation,
                   pybind11::handle noise, pybind11::handle bias);
    pybind11::array_t<double> noiseless_biasfree_measurement(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude) const;
    pybind11::array_t<double> noiseless_biased_measurement(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude) const;
    pybind11::array_t<double> noisy_biasfree_measurement(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude);
    pybind11::array_t<double> noisy_biased_measurement(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude);
    pybind11::array_t<double> jacobian_relative_quaternion(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude,pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_mrp(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude,pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_angle_vector(pybind11::handle acceleration,pybind11::handle gravity,pybind11::handle attitude,pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_bias() const;
    bool availability() const;
    V position{}; Q orientation{}; Q misalignment{}; double saturation_value;
    M noise{}; M noise_cholesky{}; V bias{}; bool enabled=true;
    std::uint64_t seed=0, random_tick=0;
private:
    pybind11::array_t<double> measure(pybind11::handle acceleration,pybind11::handle gravity,
                                      pybind11::handle attitude,bool with_bias,bool with_noise);
    V specific_force(pybind11::handle acceleration,pybind11::handle gravity) const;
};
}
#endif

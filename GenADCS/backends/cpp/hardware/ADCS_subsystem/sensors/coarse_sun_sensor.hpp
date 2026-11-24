#ifndef GENADCS_HARDWARE_SENSORS_COARSE_SUN_SENSOR_HPP
#define GENADCS_HARDWARE_SENSORS_COARSE_SUN_SENSOR_HPP
#include <pybind11/numpy.h>
#include <array>
#include <cstdint>
namespace omni::hardware::sensors {
struct CoarseSunSensor {
    using V=std::array<double,3>; using Q=std::array<double,4>;
    CoarseSunSensor(pybind11::handle position,pybind11::handle orientation,pybind11::handle misalignment,
        pybind11::handle normal,double sensitivity,double saturation,double noise_variance,
        double bias,double fov_half_angle);
    double noiseless_biasfree_measurement(pybind11::handle sun,pybind11::handle attitude) const;
    double noiseless_biased_measurement(pybind11::handle sun,pybind11::handle attitude) const;
    double noisy_biasfree_measurement(pybind11::handle sun,pybind11::handle attitude);
    double noisy_biased_measurement(pybind11::handle sun,pybind11::handle attitude);
    double jacobian_bias() const;
    pybind11::array_t<double> jacobian_relative_quaternion(pybind11::handle sun,pybind11::handle attitude,pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_mrp(pybind11::handle sun,pybind11::handle attitude,pybind11::handle delta) const;
    pybind11::array_t<double> jacobian_relative_angle_vector(pybind11::handle sun,pybind11::handle attitude,pybind11::handle delta) const;
    bool availability(pybind11::handle sun,pybind11::handle attitude,bool visible=true) const;
    pybind11::array_t<double> fisher_information_matrix(pybind11::handle sun,pybind11::handle attitude,bool visible=true) const;
    V position{},normal{}; Q orientation{},misalignment{};
    double sensitivity,saturation_value,noise_variance,noise_std,bias,fov_half_angle;
    bool enabled=true; std::uint64_t seed=0,random_tick=0;
private:
    double measure(pybind11::handle sun,pybind11::handle attitude,bool with_bias,bool with_noise);
};
}
#endif

#ifndef GENADCS_HARDWARE_SENSORS_ACCELEROMETER_HPP
#define GENADCS_HARDWARE_SENSORS_ACCELEROMETER_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
struct Accelerometer {
    using V = std::array<double, 3>;
    using Q = std::array<double, 4>;
    using M = std::array<std::array<double, 3>, 3>;

    Accelerometer(V position, Q orientation, Q misalignment, double saturation,
                  M noise, V bias);
    V noiseless_biasfree_measurement(const V& acceleration, const V& gravity,
                                     const Q& attitude) const;
    V noiseless_biased_measurement(const V& acceleration, const V& gravity,
                                   const Q& attitude) const;
    V noisy_biasfree_measurement(const V& acceleration, const V& gravity,
                                 const Q& attitude);
    V noisy_biased_measurement(const V& acceleration, const V& gravity,
                               const Q& attitude);
    M jacobian_relative_quaternion(const V& acceleration, const V& gravity,
                                   const Q& attitude, const Q& delta) const;
    M jacobian_relative_mrp(const V& acceleration, const V& gravity,
                            const Q& attitude, const V& delta) const;
    M jacobian_relative_angle_vector(const V& acceleration, const V& gravity,
                                     const Q& attitude, const V& delta) const;
    M jacobian_bias() const;
    bool availability() const;

    V position{};
    Q orientation{};
    Q misalignment{};
    double saturation_value;
    M noise{};
    M noise_cholesky{};
    V bias{};
    bool enabled = true;
    std::uint64_t seed = 0;
    std::uint64_t random_tick = 0;

private:
    V measure(const V& acceleration, const V& gravity, const Q& attitude,
              bool with_bias, bool with_noise);
    V specific_force(const V& acceleration, const V& gravity) const;
};
}

#endif

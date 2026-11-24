#ifndef GENADCS_HARDWARE_SENSORS_GNSS_HPP
#define GENADCS_HARDWARE_SENSORS_GNSS_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors
{
struct GNSS
{
    using V = std::array<double, 3>;
    using V6 = std::array<double, 6>;
    using M3 = std::array<std::array<double, 3>, 3>;
    using M6 = std::array<std::array<double, 6>, 6>;
    using M63 = std::array<std::array<double, 3>, 6>;

    GNSS(M6 noise, V6 bias);
    V6 noiseless_biasfree_measurement(const V &position_inertial, const V &velocity_inertial) const;
    V6 noiseless_biased_measurement(const V &position_inertial, const V &velocity_inertial) const;
    V6 noisy_biasfree_measurement(const V &position_inertial, const V &velocity_inertial);
    V6 noisy_biased_measurement(const V &position_inertial, const V &velocity_inertial);
    M63 jacobian_relative_quaternion() const;
    M63 jacobian_relative_mrp() const;
    M63 jacobian_relative_angle_vector() const;
    M63 jacobian_position() const;
    M63 jacobian_velocity() const;
    M6 jacobian_bias() const;
    bool availability() const;

    M6 noise{}, noise_cholesky{};
    V6 bias{};
    bool enabled = true;
    std::uint64_t seed = 0, random_tick = 0;

  private:
    V6 measure(const V &position_inertial, const V &velocity_inertial, bool with_bias, bool with_noise);
};
} // namespace omni::hardware::sensors
#endif

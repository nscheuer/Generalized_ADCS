#ifndef GENADCS_HARDWARE_SENSORS_GYROSCOPE_HPP
#define GENADCS_HARDWARE_SENSORS_GYROSCOPE_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
struct Gyroscope {
  using V = std::array<double, 3>;
  using Q = std::array<double, 4>;
  using M = std::array<std::array<double, 3>, 3>;

  Gyroscope(V position, Q orientation, Q misalignment, double saturation_value,
            M noise, V bias);
  V noiseless_biasfree_measurement(const V &rate) const;
  V noiseless_biased_measurement(const V &rate) const;
  V noisy_biasfree_measurement(const V &rate);
  V noisy_biased_measurement(const V &rate);
  M jacobian_angular_velocity() const;
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
  V measure(const V &rate, bool with_bias, bool with_noise);
};
} // namespace omni::hardware::sensors

#endif

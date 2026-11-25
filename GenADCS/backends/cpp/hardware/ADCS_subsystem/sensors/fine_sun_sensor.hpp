#ifndef GENADCS_HARDWARE_SENSORS_FINE_SUN_SENSOR_HPP
#define GENADCS_HARDWARE_SENSORS_FINE_SUN_SENSOR_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
struct FineSunSensor {
  using V = std::array<double, 3>;
  using Q = std::array<double, 4>;
  using V2 = std::array<double, 2>;
  using M2 = std::array<std::array<double, 2>, 2>;
  using M23 = std::array<std::array<double, 3>, 2>;
  using M3 = std::array<std::array<double, 3>, 3>;

  FineSunSensor(V position, Q orientation, Q misalignment, M2 noise, V2 bias,
                V2 fov_half_angles);
  V2 noiseless_biasfree_measurement(const V &sun, const Q &attitude) const;
  V2 noiseless_biased_measurement(const V &sun, const Q &attitude) const;
  V2 noisy_biasfree_measurement(const V &sun, const Q &attitude);
  V2 noisy_biased_measurement(const V &sun, const Q &attitude);
  M2 jacobian_bias() const;
  M23 jacobian_relative_quaternion(const V &sun, const Q &attitude,
                                   const Q &delta) const;
  M23 jacobian_relative_mrp(const V &sun, const Q &attitude,
                            const V &delta) const;
  M23 jacobian_relative_angle_vector(const V &sun, const Q &attitude,
                                     const V &delta) const;
  bool availability(const V &sun, const Q &attitude, bool visible = true) const;
  M3 fisher_information_matrix(const V &sun, const Q &attitude,
                               bool visible = true) const;

  V position{};
  Q orientation{}, misalignment{};
  M2 noise{}, noise_cholesky{};
  V2 bias{}, fov_half_angles{};
  bool enabled = true;
  std::uint64_t seed = 0, random_tick = 0;

private:
  V2 measure(const V &sun, const Q &attitude, bool with_bias, bool with_noise);
};
} // namespace omni::hardware::sensors
#endif

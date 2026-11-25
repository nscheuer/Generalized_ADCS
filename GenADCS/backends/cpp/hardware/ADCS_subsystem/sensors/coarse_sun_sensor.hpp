#ifndef GENADCS_HARDWARE_SENSORS_COARSE_SUN_SENSOR_HPP
#define GENADCS_HARDWARE_SENSORS_COARSE_SUN_SENSOR_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
struct CoarseSunSensor {
  using V = std::array<double, 3>;
  using Q = std::array<double, 4>;
  using M = std::array<std::array<double, 3>, 3>;

  CoarseSunSensor(V position, Q orientation, Q misalignment, V normal,
                  double sensitivity, double saturation, double noise_variance,
                  double bias, double fov_half_angle);
  double noiseless_biasfree_measurement(const V &sun, const Q &attitude) const;
  double noiseless_biased_measurement(const V &sun, const Q &attitude) const;
  double noisy_biasfree_measurement(const V &sun, const Q &attitude);
  double noisy_biased_measurement(const V &sun, const Q &attitude);
  double jacobian_bias() const;
  V jacobian_relative_quaternion(const V &sun, const Q &attitude,
                                 const Q &delta) const;
  V jacobian_relative_mrp(const V &sun, const Q &attitude,
                          const V &delta) const;
  V jacobian_relative_angle_vector(const V &sun, const Q &attitude,
                                   const V &delta) const;
  bool availability(const V &sun, const Q &attitude, bool visible = true) const;
  M fisher_information_matrix(const V &sun, const Q &attitude,
                              bool visible = true) const;

  V position{}, normal{};
  Q orientation{}, misalignment{};
  double sensitivity, saturation_value, noise_variance, noise_std, bias,
      fov_half_angle;
  bool enabled = true;
  std::uint64_t seed = 0, random_tick = 0;

private:
  double measure(const V &sun, const Q &attitude, bool with_bias,
                 bool with_noise);
};
} // namespace omni::hardware::sensors
#endif

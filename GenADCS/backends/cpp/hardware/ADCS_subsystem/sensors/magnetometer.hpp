#ifndef GENADCS_HARDWARE_SENSORS_MAGNETOMETER_HPP
#define GENADCS_HARDWARE_SENSORS_MAGNETOMETER_HPP

#include <array>
#include <cstdint>

namespace omni::hardware::sensors {
struct Magnetometer {
  using V = std::array<double, 3>;
  using Q = std::array<double, 4>;
  using M = std::array<std::array<double, 3>, 3>;

  Magnetometer(V position, Q orientation, Q misalignment,
               double saturation_value, M noise, V bias);
  V noiseless_biasfree_measurement(const V &field, const Q &attitude) const;
  V noiseless_biased_measurement(const V &field, const Q &attitude) const;
  V noisy_biasfree_measurement(const V &field, const Q &attitude);
  V noisy_biased_measurement(const V &field, const Q &attitude);
  M jacobian_bias() const;
  M jacobian_relative_quaternion(const V &field, const Q &attitude,
                                 const Q &delta) const;
  M jacobian_relative_mrp(const V &field, const Q &attitude,
                          const V &delta) const;
  M jacobian_relative_angle_vector(const V &field, const Q &attitude,
                                   const V &delta) const;
  bool availability() const;
  M fisher_information_matrix(const V &field, const Q &attitude,
                              const M &field_covariance) const;

  V position{};
  Q orientation{}, misalignment{};
  double saturation_value;
  M noise{}, noise_cholesky{};
  V bias{};
  bool enabled = true;
  std::uint64_t seed = 0, random_tick = 0;

private:
  V measure(const V &field, const Q &attitude, bool with_bias, bool with_noise);
  V sensor_field(const V &field, const Q &attitude) const;
};
} // namespace omni::hardware::sensors

#endif

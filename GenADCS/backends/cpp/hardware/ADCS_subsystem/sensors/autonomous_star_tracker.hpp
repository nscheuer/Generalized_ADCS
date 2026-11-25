#ifndef GENADCS_HARDWARE_SENSORS_AUTONOMOUS_STAR_TRACKER_HPP
#define GENADCS_HARDWARE_SENSORS_AUTONOMOUS_STAR_TRACKER_HPP

#include <array>
#include <cstdint>
#include <optional>

namespace omni::hardware::sensors {
struct AutonomousStarTracker {
  using V = std::array<double, 3>;
  using Q = std::array<double, 4>;
  using M = std::array<std::array<double, 3>, 3>;

  AutonomousStarTracker(V position, Q orientation, Q misalignment,
                        double saturation_value, M noise, V bias,
                        double fov_half_angle = 3.14159265358979323846);
  Q noiseless_biasfree_measurement(const Q &attitude) const;
  Q noiseless_biased_measurement(const Q &attitude) const;
  Q noisy_biasfree_measurement(const Q &attitude);
  Q noisy_biased_measurement(const Q &attitude);
  V residual(const Q &measured_attitude, const Q &estimated_attitude) const;
  M jacobian_relative_quaternion() const;
  M jacobian_relative_mrp(const V &delta_mrp) const;
  M jacobian_relative_angle_vector(const V &delta_angle) const;
  bool availability(std::optional<Q> attitude = std::nullopt,
                    std::optional<V> earth_inertial = std::nullopt,
                    std::optional<V> sun_inertial = std::nullopt) const;
  M fisher_information_matrix() const;

  V position{}, bias{};
  Q orientation{}, misalignment{};
  M noise{}, noise_cholesky{};
  double saturation_value, fov_half_angle;
  bool enabled = true;
  std::uint64_t seed = 0, random_tick = 0;
};
} // namespace omni::hardware::sensors
#endif

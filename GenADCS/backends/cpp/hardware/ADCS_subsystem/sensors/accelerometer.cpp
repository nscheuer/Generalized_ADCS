#include "accelerometer.hpp"
#include "kernels.hpp"
#include "philox.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::sensors {
Accelerometer::Accelerometer(V p, Q q, Q m, double saturation, M covariance,
                             V b)
    : position(p), orientation(kernels::normalize(q)),
      misalignment(kernels::normalize(m)), saturation_value(saturation),
      noise(covariance), noise_cholesky(kernels::cholesky(noise)), bias(b) {
  if (!(saturation_value > 0.0)) {
    throw std::invalid_argument("saturation_value must be positive");
}
}

Accelerometer::V Accelerometer::specific_force(const V &acceleration,
                                               const V &gravity) const {
  V force{};
  for (std::size_t i = 0; i < force.size(); ++i) {
    force[i] = acceleration[i] - gravity[i];
}
  return force;
}

Accelerometer::V Accelerometer::measure(const V &acceleration, const V &gravity,
                                        const Q &attitude, bool with_bias,
                                        bool with_noise) {
  V draw{};
  if (with_noise) {
    const auto values = omni::helpers::random::standard_normals(
        seed, random_tick++, 0x40000000U, 3);
    for (std::size_t i = 0; i < draw.size(); ++i) {
      draw[i] = values[i];
}
  }
  return kernels::accelerometer_measurement(
      acceleration, gravity, attitude, orientation, misalignment, bias,
      noise_cholesky, saturation_value, with_bias, with_noise,
      with_noise ? &draw : nullptr);
}

Accelerometer::V
Accelerometer::noiseless_biasfree_measurement(const V &a, const V &g,
                                              const Q &q) const {
  return const_cast<Accelerometer *>(this)->measure(a, g, q, false, false);
}
Accelerometer::V Accelerometer::noiseless_biased_measurement(const V &a,
                                                             const V &g,
                                                             const Q &q) const {
  return const_cast<Accelerometer *>(this)->measure(a, g, q, true, false);
}
Accelerometer::V
Accelerometer::noisy_biasfree_measurement(const V &a, const V &g, const Q &q) {
  return measure(a, g, q, false, true);
}
Accelerometer::V Accelerometer::noisy_biased_measurement(const V &a, const V &g,
                                                         const Q &q) {
  return measure(a, g, q, true, true);
}

Accelerometer::M
Accelerometer::jacobian_relative_quaternion(const V &a, const V &g, const Q &q,
                                            const Q &delta) const {
  const V force = kernels::rotate_inertial_to_sensor(specific_force(a, g), q,
                                                     orientation, misalignment);
  return kernels::attitude_vector_jacobian_quaternion(force, delta);
}
Accelerometer::M Accelerometer::jacobian_relative_mrp(const V &a, const V &g,
                                                      const Q &q,
                                                      const V &delta) const {
  const V force = kernels::rotate_inertial_to_sensor(specific_force(a, g), q,
                                                     orientation, misalignment);
  return kernels::attitude_vector_jacobian_mrp(force, delta);
}
Accelerometer::M Accelerometer::jacobian_relative_angle_vector(
    const V &a, const V &g, const Q &q, const V &delta) const {
  const V force = kernels::rotate_inertial_to_sensor(specific_force(a, g), q,
                                                     orientation, misalignment);
  return kernels::attitude_vector_jacobian_angle(force, delta);
}
Accelerometer::M Accelerometer::jacobian_bias() const {
  return kernels::identity();
}
bool Accelerometer::availability() const { return enabled; }
} // namespace omni::hardware::sensors

#include "coarse_sun_sensor.hpp"
#include "kernels.hpp"
#include "philox.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::sensors {
CoarseSunSensor::CoarseSunSensor(V p, Q q, Q m, V n, double sens, double sat,
                                 double var, double b, double fov)
    : position(p), normal(n), orientation(kernels::normalize(q)),
      misalignment(kernels::normalize(m)), sensitivity(sens),
      saturation_value(sat), noise_variance(var), noise_std(std::sqrt(var)),
      bias(b), fov_half_angle(fov) {
  const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                normal[2] * normal[2]);
  if (norm == 0 || sensitivity <= 0 || saturation_value <= 0) {
    throw std::invalid_argument(
        "normal must be nonzero; sensitivity and saturation positive");
  }
  if (noise_variance <= 0 || fov_half_angle <= 0 ||
      fov_half_angle > std::acos(-1.0) / 2) {
    throw std::invalid_argument(
        "noise variance and FOV must be positive; FOV <= pi/2");
  }
  for (double &x : normal) {
    x /= norm;
  }
}

double CoarseSunSensor::measure(const V &sun, const Q &attitude, bool with_bias,
                                bool with_noise) {
  double draw = 0;
  if (with_noise) {
    const auto values = omni::helpers::random::standard_normals(
        seed, random_tick++, 0x40000000U, 1);
    draw = values[0];
  }
  return kernels::scalar_sun_measurement(
      sun, attitude, orientation, misalignment, normal, sensitivity, bias,
      noise_std, saturation_value, with_bias, with_noise, draw);
}

double CoarseSunSensor::noiseless_biasfree_measurement(const V &s,
                                                       const Q &q) const {
  return const_cast<CoarseSunSensor *>(this)->measure(s, q, false, false);
}
double CoarseSunSensor::noiseless_biased_measurement(const V &s,
                                                     const Q &q) const {
  return const_cast<CoarseSunSensor *>(this)->measure(s, q, true, false);
}
double CoarseSunSensor::noisy_biasfree_measurement(const V &s, const Q &q) {
  return measure(s, q, false, true);
}
double CoarseSunSensor::noisy_biased_measurement(const V &s, const Q &q) {
  return measure(s, q, true, true);
}
double CoarseSunSensor::jacobian_bias() const { return 1.0; }

CoarseSunSensor::V
CoarseSunSensor::jacobian_relative_quaternion(const V &s, const Q &q,
                                              const Q &d) const {
  return kernels::coarse_sun_attitude_jacobian(s, q, orientation, misalignment,
                                               normal, sensitivity, d, 0);
}
CoarseSunSensor::V CoarseSunSensor::jacobian_relative_mrp(const V &s,
                                                          const Q &q,
                                                          const V &d) const {
  return kernels::coarse_sun_attitude_jacobian(s, q, orientation, misalignment,
                                               normal, sensitivity, d, 1);
}
CoarseSunSensor::V
CoarseSunSensor::jacobian_relative_angle_vector(const V &s, const Q &q,
                                                const V &d) const {
  return kernels::coarse_sun_attitude_jacobian(s, q, orientation, misalignment,
                                               normal, sensitivity, d, 2);
}
bool CoarseSunSensor::availability(const V &s, const Q &q, bool visible) const {
  return kernels::coarse_sun_available(s, q, orientation, misalignment, normal,
                                       fov_half_angle, enabled, visible);
}
CoarseSunSensor::M
CoarseSunSensor::fisher_information_matrix(const V &s, const Q &q,
                                           bool visible) const {
  if (availability(s, q, visible)) {
    return kernels::coarse_sun_information(s, q, orientation, misalignment,
                                           normal, sensitivity, noise_variance);
  }
  return {};
}
} // namespace omni::hardware::sensors

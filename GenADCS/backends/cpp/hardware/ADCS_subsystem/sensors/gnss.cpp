#include "gnss.hpp"
#include "philox.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::sensors {
GNSS::GNSS(M6 covariance, V6 b) : noise(covariance), bias(b) {
  for (std::size_t i = 0; i < 6; ++i) {
    if (std::abs(noise[i][i]) <= 0.0) {
      throw std::invalid_argument("noise covariance must be positive definite");
}
    for (std::size_t j = 0; j < i; ++j) {
      if (std::abs(noise[i][j] - noise[j][i]) >
          1e-12 * (1.0 + std::abs(noise[i][j]))) {
        throw std::invalid_argument(
            "noise covariance must be symmetric positive definite");
}
}
  }
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double value = 0.5 * (noise[i][j] + noise[j][i]);
      for (std::size_t k = 0; k < j; ++k) {
        value -= noise_cholesky[i][k] * noise_cholesky[j][k];
}
      if (i == j) {
        if (!(value > 0.0)) {
          throw std::invalid_argument(
              "noise covariance must be positive definite");
}
        noise_cholesky[i][j] = std::sqrt(value);
      } else {
        noise_cholesky[i][j] = value / noise_cholesky[j][j];
}
    }
  }
}

GNSS::V6 GNSS::measure(const V &p, const V &v, bool with_bias,
                       bool with_noise) {
  V6 result{};
  for (std::size_t i = 0; i < 3; ++i) {
    result[i] = p[i];
    result[i + 3] = v[i];
  }
  if (with_bias) {
    for (std::size_t i = 0; i < 6; ++i) {
      result[i] += bias[i];
}
}
  if (with_noise) {
    const auto draw = omni::helpers::random::standard_normals(
        seed, random_tick++, 0x40000000U, 6);
    for (std::size_t i = 0; i < 6; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        result[i] += noise_cholesky[i][j] * draw[j];
}
}
  }
  return result;
}
GNSS::V6 GNSS::noiseless_biasfree_measurement(const V &p, const V &v) const {
  return const_cast<GNSS *>(this)->measure(p, v, false, false);
}
GNSS::V6 GNSS::noiseless_biased_measurement(const V &p, const V &v) const {
  return const_cast<GNSS *>(this)->measure(p, v, true, false);
}
GNSS::V6 GNSS::noisy_biasfree_measurement(const V &p, const V &v) {
  return measure(p, v, false, true);
}
GNSS::V6 GNSS::noisy_biased_measurement(const V &p, const V &v) {
  return measure(p, v, true, true);
}
GNSS::M63 GNSS::jacobian_relative_quaternion() const { return {}; }
GNSS::M63 GNSS::jacobian_relative_mrp() const { return {}; }
GNSS::M63 GNSS::jacobian_relative_angle_vector() const { return {}; }
GNSS::M63 GNSS::jacobian_position() const {
  M63 r{};
  for (std::size_t i = 0; i < 3; ++i) {
    r[i][i] = 1;
}
  return r;
}
GNSS::M63 GNSS::jacobian_velocity() const {
  M63 r{};
  for (std::size_t i = 0; i < 3; ++i) {
    r[i + 3][i] = 1;
}
  return r;
}
GNSS::M6 GNSS::jacobian_bias() const {
  M6 r{};
  for (std::size_t i = 0; i < 6; ++i) {
    r[i][i] = 1;
}
  return r;
}
bool GNSS::availability() const { return enabled; }
} // namespace omni::hardware::sensors

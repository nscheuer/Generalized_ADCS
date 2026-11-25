#include "magnetometer.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::Magnetometer;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)

namespace {
const Magnetometer::Q identity{{1, 0, 0, 0}};
Magnetometer::M diagonal(double x, double y, double z) {
  return {{{{x, 0, 0}}, {{0, y, 0}}, {{0, 0, z}}}};
}
Magnetometer make_sensor(double limit = 1.,
                         Magnetometer::M noise = diagonal(1e-12, 1e-12, 1e-12),
                         Magnetometer::V bias = {{0, 0, 0}},
                         Magnetometer::Q orientation = identity,
                         Magnetometer::Q misalignment = identity) {
  return {{{0, 0, 0}}, orientation, misalignment, limit, noise, bias};
}
void measurements() {
  auto s = make_sensor(3e-5);
  auto result =
      s.noiseless_biasfree_measurement({{5e-5, -4e-5, 1e-5}}, identity);
  CHECK(std::abs(result[0] - 3e-5) < 1e-15 &&
        std::abs(result[1] + 3e-5) < 1e-15 &&
        std::abs(result[2] - 1e-5) < 1e-15);
  auto biased = make_sensor(4e-5, diagonal(1, 1, 1), {{1e-5, -2e-5, 0}});
  result = biased.noiseless_biased_measurement({{4e-5, -4e-5, 2e-5}}, identity);
  CHECK(std::abs(result[0] - 4e-5) < 1e-15 &&
        std::abs(result[1] + 4e-5) < 1e-15 &&
        std::abs(result[2] - 2e-5) < 1e-15);
  const double h = std::sqrt(.5);
  auto rotated = make_sensor(1., diagonal(1, 1, 1), {{0, 0, 0}}, {{h, h, 0, 0}},
                             {{h, 0, h, 0}});
  result = rotated.noiseless_biasfree_measurement({{2e-5, -3e-5, 5e-5}},
                                                  {{h, 0, 0, h}});
  const double expected[] = {2e-5, -5e-5, -3e-5};
  for (int i = 0; i < 3; ++i)
    CHECK(std::abs(result[i] - expected[i]) < 1e-14);
  const auto jac = biased.jacobian_bias();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      CHECK(jac[i][j] == (i == j ? 1. : 0.));
}
void noise_jacobians_and_information() {
  auto s = make_sensor(1., diagonal(1e-10, 4e-10, 9e-10));
  s.seed = 12;
  auto noisy = s.noisy_biasfree_measurement({{0, 0, 0}}, identity);
  for (double value : noisy)
    CHECK(std::isfinite(value));
  const Magnetometer::V field{{2e-5, -3e-5, 4e-5}};
  auto qj = s.jacobian_relative_quaternion(field, identity, identity);
  CHECK(std::abs(qj[0][1] - 8e-5) < 1e-14 && std::abs(qj[1][0] + 8e-5) < 1e-14);
  const auto mj = s.jacobian_relative_mrp(field, identity, {{0, 0, 0}});
  const auto aj =
      s.jacobian_relative_angle_vector(field, identity, {{0, 0, 0}});
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      CHECK(std::isfinite(mj[i][j]) && std::isfinite(aj[i][j]));
      CHECK(std::abs(mj[i][j] - 2 * qj[i][j]) < 1e-14);
      CHECK(std::abs(aj[i][j] - .5 * qj[i][j]) < 1e-14);
    }
  auto info_sensor = make_sensor(1., diagonal(2, 3, 4));
  const Magnetometer::M covariance = diagonal(.1, .2, .3);
  const auto info =
      info_sensor.fisher_information_matrix({{1, 2, 3}}, identity, covariance);
  CHECK(std::abs(info[0][0] - (9 / 3.2 + 4 / 4.3)) < 1e-12);
  CHECK(info_sensor.availability());
  info_sensor.enabled = false;
  CHECK(!info_sensor.availability());
  const auto zero =
      info_sensor.fisher_information_matrix({{1, 2, 3}}, identity, covariance);
  for (const auto &row : zero)
    for (double value : row)
      CHECK(value == 0.);
}
void invalid_saturation_rejected() {
  bool rejected = false;
  try {
    make_sensor(0.);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}
} // namespace
int main() {
  measurements();
  noise_jacobians_and_information();
  invalid_saturation_rejected();
}

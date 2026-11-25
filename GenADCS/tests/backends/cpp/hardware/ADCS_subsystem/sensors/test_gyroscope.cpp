#include "gyroscope.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::Gyroscope;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)

namespace {
const Gyroscope::Q identity{{1, 0, 0, 0}};
Gyroscope make_sensor(
    double limit = 100.0,
    Gyroscope::M noise = {{{{1e-8, 0, 0}}, {{0, 1e-8, 0}}, {{0, 0, 1e-8}}}},
    Gyroscope::V bias = {{0, 0, 0}}, Gyroscope::Q orientation = identity,
    Gyroscope::Q misalignment = identity) {
  return {{{0, 0, 0}}, orientation, misalignment, limit, noise, bias};
}

void mounting_bias_and_saturation() {
  const double half = std::sqrt(.5);
  auto rotated =
      make_sensor(100, {{{{1e-8, 0, 0}}, {{0, 1e-8, 0}}, {{0, 0, 1e-8}}}},
                  {{0, 0, 0}}, {{half, 0, 0, half}}, {{half, half, 0, 0}});
  const auto result = rotated.noiseless_biasfree_measurement({{.1, -.2, .3}});
  CHECK(std::abs(result[0] - .2) < 1e-12);
  CHECK(std::abs(result[1] + .3) < 1e-12);
  CHECK(std::abs(result[2] - .1) < 1e-12);

  auto biased =
      make_sensor(100, {{{{1e-8, 0, 0}}, {{0, 1e-8, 0}}, {{0, 0, 1e-8}}}},
                  {{.01, -.02, .03}});
  const auto with_bias = biased.noiseless_biased_measurement({{.1, .2, -.3}});
  CHECK(std::abs(with_bias[0] - .11) < 1e-12);
  CHECK(std::abs(with_bias[1] - .18) < 1e-12);
  CHECK(std::abs(with_bias[2] + .27) < 1e-12);
  const auto jb = biased.jacobian_bias();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      CHECK(jb[i][j] == (i == j ? 1. : 0.));

  auto saturated = make_sensor(.25);
  const auto clipped =
      saturated.noiseless_biasfree_measurement({{.5, -.4, .1}});
  CHECK(clipped[0] == .25 && clipped[1] == -.25 && clipped[2] == .1);
}

void noisy_measurements_and_api_contract() {
  auto sensor =
      make_sensor(100, {{{{1e-6, 0, 0}}, {{0, 4e-6, 0}}, {{0, 0, 9e-6}}}},
                  {{.01, 0, -.01}});
  sensor.seed = 31;
  for (const auto value : {sensor.noisy_biasfree_measurement({{.1, .2, .3}}),
                           sensor.noisy_biased_measurement({{.1, .2, .3}})})
    for (double component : value)
      CHECK(std::isfinite(component));
  const auto j = sensor.jacobian_angular_velocity();
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      CHECK(j[i][k] == (i == k ? 1. : 0.));
  CHECK(sensor.availability());
  sensor.enabled = false;
  CHECK(!sensor.availability());
}

void rejects_invalid_configuration() {
  bool rejected = false;
  try {
    make_sensor(0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    make_sensor(10, {{{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}}}});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}
} // namespace

int main() {
  mounting_bias_and_saturation();
  noisy_measurements_and_api_contract();
  rejects_invalid_configuration();
}

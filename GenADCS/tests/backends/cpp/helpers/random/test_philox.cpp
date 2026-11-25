#include "philox.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)

namespace random_helper = omni::helpers::random;

void test_philox4x32_10_reference_vector() {
  const auto words = random_helper::philox4x32(0, 0, 0, 0);
  CHECK(words[0] == 0x6627E8D5U);
  CHECK(words[1] == 0xE169C58DU);
  CHECK(words[2] == 0xBC57AC4CU);
  CHECK(words[3] == 0x9B00DBD8U);
}

void test_normals_are_repeatable_and_counter_separated() {
  constexpr std::uint64_t seed = 123456789;
  const auto values = random_helper::standard_normals(seed, 7, 2, 6);
  const auto repeated = random_helper::standard_normals(seed, 7, 2, 6);
  const auto other_tick = random_helper::standard_normals(seed, 8, 2, 6);
  const auto other_sensor = random_helper::standard_normals(seed, 7, 3, 6);
  const auto other_seed = random_helper::standard_normals(seed + 1, 7, 2, 6);
  CHECK(values == repeated);
  CHECK(values != other_tick);
  CHECK(values != other_sensor);
  CHECK(values != other_seed);
  for (double value : values)
    CHECK(std::isfinite(value));
}

void test_invalid_seed_is_rejected() {
  const auto rejected = [](auto seed) {
    try {
      (void)random_helper::validate_seed(seed);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  CHECK(rejected(-1LL));
  CHECK(rejected(std::ldexp(1.0, 64)));
  CHECK(rejected(1.5));
  CHECK(rejected(true));
  CHECK(
      random_helper::validate_seed(std::numeric_limits<std::uint64_t>::max()) ==
      std::numeric_limits<std::uint64_t>::max());
}

int main() {
  test_philox4x32_10_reference_vector();
  test_normals_are_repeatable_and_counter_separated();
  test_invalid_seed_is_rejected();
}

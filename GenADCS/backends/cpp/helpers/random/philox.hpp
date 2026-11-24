#ifndef GENADCS_HELPERS_RANDOM_PHILOX_HPP
#define GENADCS_HELPERS_RANDOM_PHILOX_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace omni::helpers::random {
std::array<std::uint32_t, 4> philox4x32(std::uint64_t seed, std::uint64_t tick,
                                       std::uint32_t sensor, std::uint32_t block);
inline std::uint64_t validate_seed(bool) {
    throw std::invalid_argument("seed must be an integer in [0, 2**64)");
}
template <typename Integer,
          std::enable_if_t<std::is_integral_v<Integer> && !std::is_same_v<Integer, bool>, int> = 0>
std::uint64_t validate_seed(Integer seed) {
    if constexpr (std::is_signed_v<Integer>) {
        if (seed < 0) throw std::invalid_argument("seed must be an integer in [0, 2**64)");
    }
    return static_cast<std::uint64_t>(seed);
}
template <typename Floating, std::enable_if_t<std::is_floating_point_v<Floating>, int> = 0>
std::uint64_t validate_seed(Floating seed) {
    const long double value = static_cast<long double>(seed);
    if (!std::isfinite(value) || value < 0.0L || std::floor(value) != value ||
        value >= 18446744073709551616.0L)
        throw std::invalid_argument("seed must be an integer in [0, 2**64)");
    return static_cast<std::uint64_t>(value);
}
std::vector<double> standard_normals(std::uint64_t seed, std::uint64_t tick,
                                     std::uint32_t sensor, std::uint32_t count);
}

#endif

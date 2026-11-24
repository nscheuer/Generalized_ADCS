#include "philox.hpp"

#include <cmath>

namespace omni::helpers::random
{
namespace
{
constexpr std::uint64_t mask32 = 0xffffffffULL;
constexpr double inverse_uint32_range = 1.0 / 4294967296.0;
constexpr double two_pi = 6.283185307179586476925286766559;
} // namespace

std::array<std::uint32_t, 4> philox4x32(std::uint64_t seed, std::uint64_t tick, std::uint32_t sensor,
                                        std::uint32_t block)
{
    // Counter words: tick low, tick high, sensor index, block. Key words come
    // from seed.
    std::uint64_t c0 = tick & mask32;
    std::uint64_t c1 = (tick >> 32U) & mask32;
    std::uint64_t c2 = sensor;
    std::uint64_t c3 = block;
    std::uint64_t k0 = seed & mask32;
    std::uint64_t k1 = (seed >> 32U) & mask32;

    for (int round = 0; round < 10; ++round)
    {
        const std::uint64_t p0 = 0xD2511F53ULL * c0;
        const std::uint64_t p1 = 0xCD9E8D57ULL * c2;
        const std::uint64_t next_c0 = ((p1 >> 32U) ^ c1 ^ k0) & mask32;
        const std::uint64_t next_c1 = p1 & mask32;
        const std::uint64_t next_c2 = ((p0 >> 32U) ^ c3 ^ k1) & mask32;
        const std::uint64_t next_c3 = p0 & mask32;
        c0 = next_c0;
        c1 = next_c1;
        c2 = next_c2;
        c3 = next_c3;
        k0 = (k0 + 0x9E3779B9ULL) & mask32;
        k1 = (k1 + 0xBB67AE85ULL) & mask32;
    }
    return {{static_cast<std::uint32_t>(c0), static_cast<std::uint32_t>(c1), static_cast<std::uint32_t>(c2),
             static_cast<std::uint32_t>(c3)}};
}

std::vector<double> standard_normals(std::uint64_t seed, std::uint64_t tick, std::uint32_t sensor, std::uint32_t count)
{
    std::vector<double> result(count);
    for (std::uint32_t block = 0; block < (count + 3U) / 4U; ++block)
    {
        const auto words = philox4x32(seed, tick, sensor, block);
        for (std::uint32_t pair = 0; pair < 2; ++pair)
        {
            const std::uint32_t offset = block * 4U + pair * 2U;
            if (offset >= count)
                break;
            const double u1 = (static_cast<double>(words[2U * pair]) + 0.5) * inverse_uint32_range;
            const double u2 = (static_cast<double>(words[2U * pair + 1U]) + 0.5) * inverse_uint32_range;
            const double radius = std::sqrt(-2.0 * std::log(u1));
            const double angle = two_pi * u2;
            result[offset] = radius * std::cos(angle);
            if (offset + 1U < count)
                result[offset + 1U] = radius * std::sin(angle);
        }
    }
    return result;
}
} // namespace omni::helpers::random

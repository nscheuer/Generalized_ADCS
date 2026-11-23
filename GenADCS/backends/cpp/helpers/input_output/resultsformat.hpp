#pragma once
#include <cstddef>

namespace omni::simulation {
struct ResultsFormat {
    bool attitude_quaternion = true;
    bool satellite_rate = true;
    bool wheel_momenta = true;
    std::size_t N = 1;
};
}

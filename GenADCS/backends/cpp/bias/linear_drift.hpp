#pragma once

#include "bias.hpp"

namespace omni::bias {
struct LinearDrift : Bias {
    explicit LinearDrift(double rate, double initial_value = 0.0)
        : rate(rate), initial_value(initial_value) {}
    double rate;
    double initial_value;
};
}  // namespace omni::bias

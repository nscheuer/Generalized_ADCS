#pragma once

#include "bias.hpp"

namespace omni::bias {
struct FixedBias : Bias {
    explicit FixedBias(double value) : value(value) {}
    double value;
};
}  // namespace omni::bias

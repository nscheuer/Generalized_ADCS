#pragma once

#include "bias.hpp"

namespace omni::bias {
struct RandomWalkBias : Bias {
    explicit RandomWalkBias(double standard_deviation, double initial_value = 0.0)
        : Bias({initial_value}, {standard_deviation}), standard_deviation(standard_deviation), initial_value(initial_value) {}
    double standard_deviation;
    double initial_value;
};
}  // namespace omni::bias

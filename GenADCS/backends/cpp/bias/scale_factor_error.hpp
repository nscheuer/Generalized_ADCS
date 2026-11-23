#pragma once

#include "bias.hpp"

namespace omni::bias {
struct ScaleFactorError : Bias {
    explicit ScaleFactorError(double fraction) : fraction(fraction) {}
    double fraction;
};
}  // namespace omni::bias

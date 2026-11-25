#pragma once

#include "bias.hpp"

namespace omni::bias {
struct ScaleFactorError : Bias {
  explicit ScaleFactorError(double fraction)
      : Bias({fraction}), fraction(fraction) {}
  double fraction;
};
} // namespace omni::bias

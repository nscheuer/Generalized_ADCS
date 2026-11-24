#pragma once
#include <functional>
#include <utility>
#include <vector>

namespace omni::hardware
{
// Effects receive the preceding result and the elapsed time. Random generators
// and other effect state belong to the callable and can be captured by
// reference.
template <class Value> using Effect = std::function<Value(Value, double)>;

template <class Value> Value apply_effects(Value value, const std::vector<Effect<Value>> &effects, double dt)
{
    for (const auto &effect : effects)
        value = effect(std::move(value), dt);
    return value;
}
} // namespace omni::hardware

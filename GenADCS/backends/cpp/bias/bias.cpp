#include "bias.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::bias
{
namespace
{
Vector expand(Vector value, std::size_t size)
{
    if (value.size() == size)
        return value;
    if (value.size() == 1)
        return Vector(size, value.front());
    throw std::invalid_argument("bias values must be scalar or have matching lengths");
}
} // namespace

Bias::Bias(Vector bias, Vector std_bias, Vector lower, Vector upper)
{
    if (bias.empty() || std_bias.empty() || lower.empty() || upper.empty())
        throw std::invalid_argument("bias values and bounds cannot be empty");
    const auto size = std::max({bias.size(), std_bias.size(), lower.size(), upper.size()});
    bias_ = expand(std::move(bias), size);
    std_bias_ = expand(std::move(std_bias), size);
    lower_ = expand(std::move(lower), size);
    upper_ = expand(std::move(upper), size);
    estimated_std_bias_.assign(size, 0.0);
    set_bounds(lower_, upper_);
}

void Bias::set_bias(Vector value)
{
    bias_ = expand(std::move(value), bias_.size());
}
void Bias::set_std_bias(Vector value)
{
    std_bias_ = expand(std::move(value), bias_.size());
}
void Bias::set_estimated_std_bias(Vector value)
{
    estimated_std_bias_ = expand(std::move(value), bias_.size());
}
void Bias::set_bounds(Vector lower, Vector upper)
{
    lower = expand(std::move(lower), bias_.size());
    upper = expand(std::move(upper), bias_.size());
    for (std::size_t i = 0; i < bias_.size(); ++i)
    {
        if (lower[i] > upper[i])
            throw std::invalid_argument("lower bound must not exceed upper bound");
    }
    lower_ = std::move(lower);
    upper_ = std::move(upper);
}

Bias::operator bool() const noexcept
{
    for (std::size_t i = 0; i < bias_.size(); ++i)
        if (bias_[i] != 0.0 || std_bias_[i] != 0.0)
            return true;
    return false;
}

Bias Bias::copy() const
{
    Bias result = *this;
    return result;
}

Bias Bias::broadcast_to(std::size_t length, const char *) const
{
    if (bias_.size() == length)
        return copy();
    if (bias_.size() != 1)
        throw std::invalid_argument("bias model length does not match output length");
    return Bias(Vector(length, bias_[0]), Vector(length, std_bias_[0]), Vector(length, lower_[0]),
                Vector(length, upper_[0]));
}

Matrix Bias::process_psd() const
{
    Matrix result(bias_.size(), Vector(bias_.size(), 0.0));
    for (std::size_t i = 0; i < bias_.size(); ++i)
        result[i][i] = std_bias_[i] * std_bias_[i];
    return result;
}

Matrix Bias::process_covariance(double dt) const
{
    if (!std::isfinite(dt) || dt < 0.0)
        throw std::invalid_argument("dt must be finite and non-negative");
    auto result = process_psd();
    for (auto &row : result)
        for (auto &value : row)
            value *= dt;
    return result;
}

void Bias::update(double j2000, double elapsed_seconds, const Vector &draw)
{
    if (!std::isfinite(j2000))
        throw std::invalid_argument("time must be finite");
    if (!std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0)
        throw std::invalid_argument("elapsed time must be finite and non-negative");
    if (draw.size() != bias_.size())
        throw std::invalid_argument("random draw length must match bias length");
    if (std::isfinite(last_bias_time_) && j2000 <= last_bias_time_)
        return;
    if (!std::isfinite(last_bias_time_))
    {
        last_bias_time_ = j2000;
        return;
    }
    const double scale = std::sqrt(elapsed_seconds);
    for (std::size_t i = 0; i < bias_.size(); ++i)
        bias_[i] = std::clamp(bias_[i] + std_bias_[i] * scale * draw[i], lower_[i], upper_[i]);
    last_bias_time_ = j2000;
    ++random_tick_;
}

Vector Bias::apply(const Vector &value) const
{
    if (value.size() != bias_.size())
        throw std::invalid_argument("value and bias must have matching shapes");
    Vector result(value.size());
    for (std::size_t i = 0; i < value.size(); ++i)
        result[i] = value[i] + bias_[i];
    return result;
}
} // namespace omni::bias

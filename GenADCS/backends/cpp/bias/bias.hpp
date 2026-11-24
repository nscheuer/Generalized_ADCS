#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace omni::bias
{

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

class Bias
{
  public:
    explicit Bias(Vector bias = {0.0}, Vector std_bias = {0.0},
                  Vector lower = {-std::numeric_limits<double>::infinity()},
                  Vector upper = {std::numeric_limits<double>::infinity()});

    const Vector &bias() const noexcept
    {
        return bias_;
    }
    const Vector &std_bias() const noexcept
    {
        return std_bias_;
    }
    const Vector &estimated_std_bias() const noexcept
    {
        return estimated_std_bias_;
    }
    const Vector &lower_bounds() const noexcept
    {
        return lower_;
    }
    const Vector &upper_bounds() const noexcept
    {
        return upper_;
    }
    double last_bias_time() const noexcept
    {
        return last_bias_time_;
    }
    void set_bias(Vector value);
    void set_std_bias(Vector value);
    void set_estimated_std_bias(Vector value);
    void set_bounds(Vector lower, Vector upper);
    void set_last_bias_time(double value) noexcept
    {
        last_bias_time_ = value;
    }

    explicit operator bool() const noexcept;
    Bias copy() const;
    Bias broadcast_to(std::size_t length, const char *owner = "sensor") const;
    Matrix process_covariance(double dt = 1.0) const;
    Matrix process_psd() const;
    void update(double j2000, double elapsed_seconds, const Vector &standard_normal_draw);
    Vector get_bias() const
    {
        return bias_;
    }
    Vector apply(const Vector &value) const;

  private:
    Vector bias_, std_bias_, estimated_std_bias_, lower_, upper_;
    double last_bias_time_ = std::numeric_limits<double>::quiet_NaN();
    std::size_t random_tick_ = 0;
};

} // namespace omni::bias

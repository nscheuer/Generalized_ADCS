#include "bias.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " #condition << '\n';                 \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (false)

using omni::bias::Bias;

int main()
{
    Bias bias({1.0, -2.0}, {0.5, 2.0}, {-1.0, -3.0}, {2.0, 3.0});
    CHECK(static_cast<bool>(bias));
    const auto covariance = bias.process_covariance(4.0);
    CHECK(covariance.size() == 2 && covariance[0].size() == 2);
    CHECK(covariance[0][0] == 1.0 && covariance[1][1] == 16.0);
    CHECK(covariance[0][1] == 0.0);
    const auto applied = bias.apply({2.0, 5.0});
    CHECK(applied[0] == 3.0 && applied[1] == 3.0);

    bias.set_last_bias_time(10.0);
    bias.update(11.0, 4.0, {1.0, -1.0});
    CHECK(bias.bias()[0] == 2.0);
    CHECK(bias.bias()[1] == -3.0); // Clamped to the configured lower bound.
    CHECK(bias.last_bias_time() == 11.0);
    bias.update(11.0, 4.0, {-1.0, 1.0});
    CHECK(bias.bias()[0] == 2.0); // Duplicate timestamps do not advance the walk.

    const auto broadcast = Bias({0.25}).broadcast_to(3);
    CHECK(broadcast.bias().size() == 3 && broadcast.bias()[2] == 0.25);
    bool rejected = false;
    try
    {
        (void)Bias({1.0, 2.0}).broadcast_to(3);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try
    {
        (void)bias.process_covariance(-1.0);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try
    {
        (void)bias.apply({1.0});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    CHECK(rejected);
    CHECK(std::abs(bias.process_psd()[1][1] - 4.0) < 1e-12);
}

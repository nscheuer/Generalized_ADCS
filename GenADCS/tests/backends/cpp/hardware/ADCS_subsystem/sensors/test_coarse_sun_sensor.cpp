#include "coarse_sun_sensor.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::CoarseSunSensor;

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    } while (false)

namespace
{
const CoarseSunSensor::Q identity{{1, 0, 0, 0}};
CoarseSunSensor make_sensor(CoarseSunSensor::V normal = {{1, 0, 0}}, double sensitivity = 2, double limit = 10,
                            double variance = .01, double bias = 0, double fov = 1.0471975511965976,
                            CoarseSunSensor::Q orientation = identity, CoarseSunSensor::Q misalignment = identity)
{
    return {{{0, 0, 0}}, orientation, misalignment, normal, sensitivity, limit, variance, bias, fov};
}

void cosine_projection_and_frontside()
{
    auto s = make_sensor();
    CHECK(std::abs(s.noiseless_biasfree_measurement({{1, 0, 0}}, identity) - 2.) < 1e-12);
    auto back = make_sensor({{1, 0, 0}}, 2, 10, .01, -.5);
    CHECK(back.noiseless_biasfree_measurement({{-1, 0, 0}}, identity) == 0.);
    CHECK(back.noiseless_biased_measurement({{-1, 0, 0}}, identity) == 0.);

    const double half = std::sqrt(.5);
    const CoarseSunSensor::Q orient{{half, 0, 0, half}};
    const CoarseSunSensor::Q mis{{std::cos(.1), std::sin(.1), 0, 0}};
    auto rotated = make_sensor({{0, 1, 0}}, 2, 10, .01, 0, 1.0471975511965976, orient, mis);
    CHECK(std::abs(rotated.noiseless_biasfree_measurement({{1, 0, 0}}, identity) - 2 * std::cos(.2)) < 1e-12);
}

void bias_saturation_and_noise()
{
    auto s = make_sensor({{1, 0, 0}}, 3, 2, 1e-8, .25);
    CHECK(s.noiseless_biasfree_measurement({{1, 0, 0}}, identity) == 2.);
    CHECK(s.noiseless_biased_measurement({{1, 0, 0}}, identity) == 2.);
    CHECK(s.jacobian_bias() == 1.);
    auto noisy = make_sensor({{1, 0, 0}}, 2, 3, 1e-8);
    noisy.seed = 17;
    const CoarseSunSensor::V sun{{.5, 0, std::sqrt(.75)}};
    for (double value :
         {noisy.noisy_biasfree_measurement(sun, identity), noisy.noisy_biased_measurement(sun, identity)})
        CHECK(std::isfinite(value) && value >= 0 && value <= 3);
}

void jacobians_availability_and_information()
{
    auto s = make_sensor({{0, 1, 0}});
    const CoarseSunSensor::V sun{{0, 0, 1}};
    const auto jq = s.jacobian_relative_quaternion(sun, identity, identity);
    const auto jm = s.jacobian_relative_mrp(sun, identity, {{0, 0, 0}});
    const auto ja = s.jacobian_relative_angle_vector(sun, identity, {{0, 0, 0}});
    const double expected_q[] = {-4, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
        CHECK(std::isfinite(jq[i]) && std::isfinite(jm[i]) && std::isfinite(ja[i]));
        CHECK(std::abs(jq[i] - expected_q[i]) < 1e-12);
        CHECK(std::abs(jm[i] - 2 * expected_q[i]) < 1e-12);
        CHECK(std::abs(ja[i] - .5 * expected_q[i]) < 1e-12);
    }
    const auto nonzero_q =
        s.jacobian_relative_quaternion({{.2, -.1, .97}}, identity, {{std::cos(.05), std::sin(.05), 0, 0}});
    const auto nonzero_m = s.jacobian_relative_mrp({{.2, -.1, .97}}, identity, {{.02, -.03, .01}});
    const auto nonzero_a = s.jacobian_relative_angle_vector({{.2, -.1, .97}}, identity, {{.02, -.03, .01}});
    for (int i = 0; i < 3; ++i)
        CHECK(std::isfinite(nonzero_q[i]) && std::isfinite(nonzero_m[i]) && std::isfinite(nonzero_a[i]));

    auto f = make_sensor({{0, 0, 1}}, 2, 10, .5, 0, .3);
    CHECK(f.availability({{0, 0, 1}}, identity));
    CHECK(!f.availability({{0, 0, 1}}, identity, false));
    CHECK(!f.availability({{0, 0, -1}}, identity));
    CHECK(!f.availability({{std::sin(.5), 0, std::cos(.5)}}, identity));
    const auto info = f.fisher_information_matrix({{0, std::sin(.2), std::cos(.2)}}, identity);
    const double expected = 8 * std::sin(.2) * std::sin(.2);
    CHECK(std::abs(info[0][0] - expected) < 1e-12);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            CHECK(std::isfinite(info[i][j]));
            if (i != 0 || j != 0)
                CHECK(std::abs(info[i][j]) < 1e-12);
        }
    const auto zero = f.fisher_information_matrix({{0, 0, 1}}, identity, false);
    for (const auto &row : zero)
        for (double value : row)
            CHECK(value == 0.);
}

void rejects_invalid_configuration()
{
    const auto rejects = [](const std::function<void()> &construct) {
        try
        {
            construct();
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };
    CHECK(rejects([] { make_sensor({{0, 0, 0}}); }));
    CHECK(rejects([] { make_sensor({{1, 0, 0}}, 0); }));
    CHECK(rejects([] { make_sensor({{1, 0, 0}}, 2, 10, .01, 0, 1.6); }));
    CHECK(rejects([] { make_sensor({{1, 0, 0}}, 2, 10, 0); }));
}
} // namespace

int main()
{
    cosine_projection_and_frontside();
    bias_saturation_and_noise();
    jacobians_availability_and_information();
    rejects_invalid_configuration();
}

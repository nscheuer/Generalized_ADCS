#include "accelerometer.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

using omni::hardware::sensors::Accelerometer;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); } while (false)

namespace {
const Accelerometer::Q identity{{1, 0, 0, 0}};
Accelerometer::M covariance(double variance = 1e-6) {
    Accelerometer::M result{};
    for (int i = 0; i < 3; ++i) result[i][i] = variance;
    return result;
}
Accelerometer make_sensor(double limit = 100, Accelerometer::V bias = {{0, 0, 0}},
                          Accelerometer::Q orientation = identity,
                          Accelerometer::Q misalignment = identity,
                          Accelerometer::M noise = covariance()) {
    return {{{0, 0, 0}}, orientation, misalignment, limit, noise, bias};
}

void specific_force_and_mounting() {
    const double c_y = std::cos(.3), s_y = std::sin(.3);
    const double c_z = std::cos(.5), s_z = std::sin(.5);
    const double c_x = std::cos(-.2), s_x = std::sin(-.2);
    const Accelerometer::Q attitude{{std::cos(.15), 0, std::sin(.15), 0}};
    const Accelerometer::Q orientation{{std::cos(.25), 0, 0, std::sin(.25)}};
    const Accelerometer::Q misalignment{{std::cos(-.1), std::sin(-.1), 0, 0}};
    auto sensor = make_sensor(100, {{0, 0, 0}}, orientation, misalignment);

    // Independent rotations: (a-g), then body y, mounting z, and sensor x.
    const double body[] = {c_y * 1.5 + s_y * 5., 2., -s_y * 1.5 + c_y * 5.};
    const double mounted[] = {c_z * body[0] - s_z * body[1],
                              s_z * body[0] + c_z * body[1], body[2]};
    const double expected[] = {mounted[0], c_x * mounted[1] - s_x * mounted[2],
                               s_x * mounted[1] + c_x * mounted[2]};
    const auto measured = sensor.noiseless_biasfree_measurement(
        {{2, 3, 4}}, {{.5, 1, -1}}, attitude);
    for (int i = 0; i < 3; ++i) CHECK(std::abs(measured[i] - expected[i]) < 1e-11);
}

void bias_noise_and_jacobians() {
    auto sensor = make_sensor(100, {{.1, -.2, .3}});
    const Accelerometer::V acceleration{{2, 3, 4}}, gravity{{1, 1, 1}};
    const auto base = sensor.noiseless_biasfree_measurement(acceleration, gravity, identity);
    const auto biased = sensor.noiseless_biased_measurement(acceleration, gravity, identity);
    const double expected_base[] = {1, 2, 3}, expected_biased[] = {1.1, 1.8, 3.3};
    for (int i = 0; i < 3; ++i) {
        CHECK(std::abs(base[i] - expected_base[i]) < 1e-12);
        CHECK(std::abs(biased[i] - expected_biased[i]) < 1e-12);
    }

    sensor.seed = 17;
    for (const auto value : {sensor.noisy_biasfree_measurement(acceleration, gravity, identity),
                             sensor.noisy_biased_measurement(acceleration, gravity, identity)})
        for (double component : value) CHECK(std::isfinite(component));

    const auto jq = sensor.jacobian_relative_quaternion(acceleration, gravity, identity, identity);
    const auto jm = sensor.jacobian_relative_mrp(acceleration, gravity, identity, {{0, 0, 0}});
    const auto ja = sensor.jacobian_relative_angle_vector(acceleration, gravity, identity, {{0, 0, 0}});
    const auto jb = sensor.jacobian_bias();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        CHECK(std::isfinite(jq[i][j]) && std::isfinite(jm[i][j]) && std::isfinite(ja[i][j]));
        CHECK(std::abs(jm[i][j] - 2 * jq[i][j]) < 1e-12);
        CHECK(std::abs(ja[i][j] - .5 * jq[i][j]) < 1e-12);
        CHECK(jb[i][j] == (i == j ? 1. : 0.));
    }
}

void nonzero_jacobians_saturation_and_availability() {
    auto sensor = make_sensor();
    const Accelerometer::V acceleration{{1, 1, 1}}, gravity{{.1, .2, .3}};
    const Accelerometer::Q delta{{std::cos(.05), std::sin(.05) / std::sqrt(5.),
                                  2 * std::sin(.05) / std::sqrt(5.), 0}};
    const auto jq = sensor.jacobian_relative_quaternion(acceleration, gravity, identity, delta);
    const auto jm = sensor.jacobian_relative_mrp(acceleration, gravity, identity, {{.01, -.02, .03}});
    const auto ja = sensor.jacobian_relative_angle_vector(acceleration, gravity, identity, {{.01, -.02, .03}});
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        CHECK(std::isfinite(jq[i][j]) && std::isfinite(jm[i][j]) && std::isfinite(ja[i][j]));

    auto limited = make_sensor(.5);
    const auto result = limited.noiseless_biasfree_measurement({{1, 1, 1}}, {{0, 0, 0}}, identity);
    for (double value : result) CHECK(value == .5);
    CHECK(limited.availability());
    limited.enabled = false;
    CHECK(!limited.availability());
}

void rejects_invalid_configuration() {
    const auto rejects = [](const std::function<void()>& construct) {
        try { construct(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(rejects([] { make_sensor(0); }));
    CHECK(rejects([] { make_sensor(100, {{0, 0, 0}}, identity, identity, covariance(0)); }));
}
}

int main() {
    specific_force_and_mounting();
    bias_noise_and_jacobians();
    nonzero_jacobians_saturation_and_availability();
    rejects_invalid_configuration();
}

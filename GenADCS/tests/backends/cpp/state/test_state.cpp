#include "attitude_state.hpp"
#include "internal_state.hpp"
#include "orbital_state.hpp"
#include "state.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error("check failed: " #condition);                                                     \
    } while (false)

using namespace omni;

int main()
{
    Quaternion q;
    q << 2.0, 0.0, 0.0, 0.0;
    Vector3 rates;
    rates << 0.01, 0.02, 0.03;
    AttitudeState attitude(q, rates);
    CHECK(std::abs(attitude.quaternion_array().norm() - 1.0) < 1e-12);
    CHECK(attitude.quaternion_array()[0] == 1.0);
    bool invalid_q = false;
    try
    {
        Quaternion zero = Quaternion::Zero();
        AttitudeState bad(zero, rates);
    }
    catch (const std::invalid_argument &)
    {
        invalid_q = true;
    }
    CHECK(invalid_q);
    bool invalid_rates = false;
    try
    {
        Vector3 bad_rates = Vector3::Constant(std::numeric_limits<double>::infinity());
        AttitudeState bad(q, bad_rates);
    }
    catch (const std::invalid_argument &)
    {
        invalid_rates = true;
    }
    CHECK(invalid_rates);

    Eigen::VectorXd momentum(2);
    momentum << 0.1, -0.2;
    InternalState internal(momentum);
    CHECK(internal.wheel_momentum.size() == 2);
    CHECK(internal.wheel_momentum[1] == -0.2);
    bool invalid_momentum = false;
    try
    {
        Eigen::VectorXd bad(1);
        bad << std::numeric_limits<double>::quiet_NaN();
        InternalState state(bad);
    }
    catch (const std::invalid_argument &)
    {
        invalid_momentum = true;
    }
    CHECK(invalid_momentum);
    WheelMomentumState wheels(momentum);
    CHECK(wheels.wheel_momentum.size() == 2);

    Vector3 position;
    position << 7e6, 0, 0;
    Vector3 velocity;
    velocity << 0, 7.5e3, 0;
    OrbitalState orbit(position, velocity, 2451545.0);
    CHECK(orbit.cartesian_position()[0] == 7e6);
    CHECK(orbit.cartesian_velocity()[1] == 7.5e3);
    bool invalid_date = false;
    try
    {
        OrbitalState bad(position, velocity, std::numeric_limits<double>::infinity());
    }
    catch (const std::invalid_argument &)
    {
        invalid_date = true;
    }
    CHECK(invalid_date);
    bool kepler_not_implemented = false;
    try
    {
        OrbitalState kep(std::nullopt, std::nullopt, 2451545.0);
        (void)kep.cartesian_position();
    }
    catch (const std::logic_error &)
    {
        kepler_not_implemented = true;
    }
    CHECK(kepler_not_implemented);

    State state(attitude, orbit, internal);
    state.check_wheel_count(2);
    bool wrong_count = false;
    try
    {
        state.check_wheel_count(3);
    }
    catch (const std::invalid_argument &)
    {
        wrong_count = true;
    }
    CHECK(wrong_count);

    // Sequential C++ work stays native: repeatedly normalize and validate.
    for (int i = 0; i < 100000; ++i)
    {
        const auto normalized = state.attitude.quaternion_array();
        CHECK(std::abs(normalized.norm() - 1.0) < 1e-12);
        state.check_wheel_count(2);
    }
    return 0;
}

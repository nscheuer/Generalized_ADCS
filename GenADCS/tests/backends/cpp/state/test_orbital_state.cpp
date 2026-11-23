#include "check.hpp"
#include "orbital_state.hpp"
#include <pybind11/embed.h>
#include <limits>

using namespace omni;

int main() {
    pybind11::scoped_interpreter interpreter{};
    Vector3 position; position << 7e6, 0, 0;
    Vector3 velocity; velocity << 0, 7.5e3, 0;
    OrbitalState orbit(position, velocity, 2451545.0);
    CHECK(orbit.cartesian_position()[0] == 7e6);
    CHECK(orbit.cartesian_velocity()[1] == 7.5e3);
    position[0] = 1;
    CHECK(orbit.cartesian_position()[0] == 7e6);
    CHECK(orbit.ephemeris.is_none());
    CHECK(orbit.body_positions.empty());
    expect_throw<std::invalid_argument>([&] { OrbitalState bad(position, velocity, std::numeric_limits<double>::infinity()); });
    Vector3 invalid = position; invalid[0] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] { OrbitalState bad(invalid, velocity, 2451545.0); });
    OrbitalState keplerian(std::nullopt, std::nullopt, 2451545.0);
    expect_throw<std::logic_error>([&] { (void)keplerian.cartesian_position(); });
    expect_throw<std::logic_error>([&] { (void)keplerian.cartesian_velocity(); });
}

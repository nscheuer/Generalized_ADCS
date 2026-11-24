#include "check.hpp"
#include "orbital_state.hpp"

#include <cmath>
#include <limits>
#include <memory>

using namespace omni;

namespace
{
class CountingEphemeris final : public EphemerisProvider
{
  public:
    Vector3 position(int body_id, double julian_date) const override
    {
        ++calls;
        Vector3 result;
        result << static_cast<double>(body_id), julian_date, -static_cast<double>(body_id);
        return result;
    }

    mutable int calls = 0;
};
} // namespace

int main()
{
    Vector3 position;
    position << 7e6, 0, 0;
    Vector3 velocity;
    velocity << 0, 7.5e3, 0;
    OrbitalState orbit(position, velocity, 2451545.0);
    CHECK(orbit.cartesian_position()[0] == 7e6);
    CHECK(orbit.cartesian_velocity()[1] == 7.5e3);
    position[0] = 1;
    CHECK(orbit.cartesian_position()[0] == 7e6);
    CHECK(!orbit.ephemeris);
    CHECK(orbit.body_positions.empty());

    expect_throw<std::invalid_argument>(
        [&] { OrbitalState bad(position, velocity, std::numeric_limits<double>::infinity()); });
    Vector3 invalid = position;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] { OrbitalState bad(invalid, velocity, 2451545.0); });
    invalid = velocity;
    invalid[1] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] { OrbitalState bad(position, invalid, 2451545.0); });

    OrbitalState keplerian(std::nullopt, std::nullopt, 2451545.0);
    expect_throw<std::logic_error>([&] { (void)keplerian.cartesian_position(); });
    expect_throw<std::logic_error>([&] { (void)keplerian.cartesian_velocity(); });
    expect_throw<std::logic_error>([&] { (void)keplerian.body_position("sun", 0); });

    auto ephemeris = std::make_shared<CountingEphemeris>();
    OrbitalState with_ephemeris(position, velocity, 2451545.0, std::nullopt, ephemeris);
    const Vector3 sun = with_ephemeris.body_position("str:sun", 0);
    CHECK(sun[0] == 0 && sun[1] == 2451545.0 && sun[2] == 0);
    CHECK(with_ephemeris.body_position("str:sun", 0) == sun);
    CHECK(ephemeris->calls == 1);
    (void)with_ephemeris.body_position("int:0", 0);
    CHECK(ephemeris->calls == 2);
    CHECK(with_ephemeris.body_positions.size() == 2);
}

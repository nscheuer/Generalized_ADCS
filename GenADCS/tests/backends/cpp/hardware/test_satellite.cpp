#include "satellite.hpp"

#include <stdexcept>
#include <string>

using omni::hardware::Satellite;
using omni::hardware::SatelliteGeometry;

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    } while (false)

int main()
{
    SatelliteGeometry geometry(12.5, {{2, 0, 0, 0, 3, 0, 0, 0, 4}});
    Satellite satellite(geometry);
    CHECK(satellite.geometry.mass == 12.5);
    CHECK(satellite.geometry.inertia[0] == 2.0);
    CHECK(satellite.geometry.inertia[4] == 3.0);
    CHECK(satellite.geometry.inertia[8] == 4.0);
    CHECK(satellite.mode == "idle");

    satellite.mode = "safe";
    CHECK(satellite.mode == "safe");

    Satellite explicit_mode(geometry, "detumbling");
    CHECK(explicit_mode.mode == "detumbling");
}

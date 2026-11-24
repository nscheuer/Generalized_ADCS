#include "satellite_geometry.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using omni::hardware::SatelliteGeometry;
using omni::hardware::SatellitePanel;

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    } while (false)

void test_defaults_and_explicit_values()
{
    SatelliteGeometry geometry(12.5, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    CHECK(geometry.mass == 12.5);
    CHECK(geometry.inertia[0] == 1 && geometry.inertia[4] == 2 && geometry.inertia[8] == 3);
    CHECK(geometry.panels.empty());
    CHECK((geometry.center_of_mass == std::array<double, 3>{0, 0, 0}));

    SatellitePanel panel({1, 2, 3}, {0, 0, 1}, {2, 1});
    SatelliteGeometry detailed(7.5, {2, 0, 0, 0, 3, 0, 0, 0, 4}, {panel}, {0.25, -0.1, 0});
    CHECK(detailed.panels.size() == 1);
    CHECK(detailed.panels.front().position[1] == 2);
    CHECK(detailed.panels.front().cd == 2.2);
    CHECK((detailed.center_of_mass == std::array<double, 3>{0.25, -0.1, 0}));
}

void test_native_mutation_and_value_ownership()
{
    SatelliteGeometry geometry(1.0, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    geometry.mass = 3.5;
    geometry.inertia[0] = 4.0;
    geometry.center_of_mass[2] = 0.3;
    geometry.panels.emplace_back(std::array<double, 3>{0, 1, 0}, std::array<double, 3>{0, 0, 1},
                                 std::array<double, 2>{3, 2});
    CHECK(geometry.mass == 3.5);
    CHECK(geometry.inertia[0] == 4.0);
    CHECK(geometry.center_of_mass[2] == 0.3);
    CHECK(geometry.panels.size() == 1 && geometry.panels[0].dimensions[0] == 3);
}

int main()
{
    try
    {
        test_defaults_and_explicit_values();
        test_native_mutation_and_value_ownership();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

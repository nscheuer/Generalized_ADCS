#include "satellite_panel.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using omni::hardware::SatellitePanel;
#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

int main() {
    try {
        SatellitePanel panel({1, -2, 3}, {0, 0.6, 0.8}, {4, 5}, 0.2, 0.3, 0.5, 2.1);
        CHECK(panel.position[0] == 1 && panel.position[1] == -2 && panel.position[2] == 3);
        CHECK(panel.orientation[1] == 0.6 && panel.dimensions[0] == 4 && panel.dimensions[1] == 5);
        CHECK(panel.eta_s == 0.2 && panel.eta_d == 0.3 && panel.eta_a == 0.5 && panel.cd == 2.1);
        panel.cd = 2.4;
        CHECK(panel.cd == 2.4);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#include "dynamics_config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

using omni::config::DynamicsConfig;

int main() {
    DynamicsConfig first;
    DynamicsConfig second;
    CHECK(first.forces.empty());
    CHECK(first.torques.empty());
    first.forces.push_back(1.0);
    first.torques.push_back(2.0);
    CHECK(second.forces.empty());
    CHECK(second.torques.empty());

    DynamicsConfig config({1.0, 2.0}, {3.0});
    CHECK((config.forces == std::vector<double>{1.0, 2.0}));
    CHECK((config.torques == std::vector<double>{3.0}));
    config.forces = {4.0};
    config.torques = {5.0, 6.0};
    CHECK((config.forces == std::vector<double>{4.0}));
    CHECK((config.torques == std::vector<double>{5.0, 6.0}));
}

#include "simulation_config.hpp"

#include <functional>
#include <stdexcept>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + " failed: " #condition); \
} while (false)

int main() {
    using omni::config::SimulationConfig;

    CHECK(SimulationConfig().backend() == "Python");
    for (const std::string backend : {"Python", "C", "Rust", "JAX"}) {
        CHECK(SimulationConfig(backend).backend() == backend);
    }

    for (const std::string backend : {"", "python", "c", "CUDA"}) {
        bool rejected = false;
        try {
            (void)SimulationConfig(backend);
        } catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()) ==
                       "backend must be Python, C, Rust, or JAX";
        }
        CHECK(rejected);
    }
}

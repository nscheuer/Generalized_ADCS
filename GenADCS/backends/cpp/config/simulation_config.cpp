#include "simulation_config.hpp"

#include <stdexcept>
#include <utility>

namespace omni::config {

SimulationConfig::SimulationConfig(std::string backend) : backend_(std::move(backend)) {
    if (backend_ != "Python" && backend_ != "C" && backend_ != "Rust" && backend_ != "JAX") {
        throw std::invalid_argument("backend must be Python, C, Rust, or JAX");
    }
}

}  // namespace omni::config

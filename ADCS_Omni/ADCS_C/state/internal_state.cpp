#include "internal_state.hpp"

#include <stdexcept>

namespace omni {
InternalState::InternalState(const Eigen::VectorXd& momentum) : wheel_momentum(momentum) {
    if (!wheel_momentum.allFinite()) {
        throw std::invalid_argument("wheel_momentum must be finite");
    }
}
}

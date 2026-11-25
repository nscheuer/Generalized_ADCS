#pragma once

#include <utility>
#include <vector>

namespace omni::config {
template <typename Force, typename Torque = Force> struct BasicDynamicsConfig {
  std::vector<Force> forces;
  std::vector<Torque> torques;

  BasicDynamicsConfig() = default;
  BasicDynamicsConfig(std::vector<Force> force_values,
                      std::vector<Torque> torque_values)
      : forces(std::move(force_values)), torques(std::move(torque_values)) {}
};

using DynamicsConfig = BasicDynamicsConfig<double>;
} // namespace omni::config

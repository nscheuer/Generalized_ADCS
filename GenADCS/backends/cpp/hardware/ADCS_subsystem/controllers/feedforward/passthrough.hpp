#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDFORWARD_PASSTHROUGH_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDFORWARD_PASSTHROUGH_HPP

#include "feedforward.hpp"

#include <Eigen/Core>
#include <tuple>

namespace omni::hardware::controllers {

std::tuple<Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
passthrough_step(const Eigen::Vector4d &q_goal, const Eigen::Vector3d &w_goal,
                 const Eigen::Vector4d &q_current);

class Passthrough final : public Feedforward {
public:
  int kind() const override { return FF_PASSTHROUGH; }
  Eigen::VectorXd to_params() const override;
};

} // namespace omni::hardware::controllers

#endif

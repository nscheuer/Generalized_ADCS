#include "passthrough.hpp"

#include <cmath>

namespace omni::hardware::controllers {

std::tuple<Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
passthrough_step(const Eigen::Vector4d &q_goal, const Eigen::Vector3d &w_goal,
                 const Eigen::Vector4d &q_current) {
  Eigen::Vector4d q_d = std::isnan(q_goal[0]) ? q_current : q_goal;
  Eigen::Vector3d w_d =
      std::isnan(w_goal[0]) ? Eigen::Vector3d::Zero() : w_goal;
  Eigen::Vector3d w_d_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d ff_torque = Eigen::Vector3d::Zero();
  return {q_d, w_d, w_d_dot, ff_torque};
}

Eigen::VectorXd Passthrough::to_params() const {
  return Eigen::VectorXd::Zero(1);
}

} // namespace omni::hardware::controllers

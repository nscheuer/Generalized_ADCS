#include "l2allocator.hpp"

namespace omni::hardware::controllers {

Eigen::VectorXd allocate_l2(const Eigen::Vector3d &tau_c,
                            const Eigen::Ref<const Eigen::MatrixXd> &a_pinv,
                            const Eigen::Ref<const Eigen::MatrixXd> &null_proj,
                            const Eigen::Ref<const Eigen::VectorXd> &u_0,
                            const Eigen::Ref<const Eigen::VectorXd> &max_torque) {
  Eigen::VectorXd u_rw = a_pinv * tau_c + null_proj * u_0;
  return saturate_wheels(std::move(u_rw), max_torque);
}

} // namespace omni::hardware::controllers

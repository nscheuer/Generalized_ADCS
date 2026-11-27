#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_L2ALLOCATOR_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_L2ALLOCATOR_HPP

#include "allocator.hpp"

#include <Eigen/Core>

namespace omni::hardware::controllers {

Eigen::VectorXd allocate_l2(const Eigen::Vector3d &tau_c,
                            const Eigen::Ref<const Eigen::MatrixXd> &a_pinv,
                            const Eigen::Ref<const Eigen::MatrixXd> &null_proj,
                            const Eigen::Ref<const Eigen::VectorXd> &u_0,
                            const Eigen::Ref<const Eigen::VectorXd> &max_torque);

class L2Allocator final : public Allocator {
public:
  int kind() const override { return ALLOC_L2; }
};

} // namespace omni::hardware::controllers

#endif

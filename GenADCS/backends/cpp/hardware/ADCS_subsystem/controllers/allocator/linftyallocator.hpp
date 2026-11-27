#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_LINFINITYALLOCATOR_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_LINFINITYALLOCATOR_HPP

#include "allocator.hpp"

#include <Eigen/Core>

namespace omni::hardware::controllers {

Eigen::VectorXd allocate_linfinity(
    const Eigen::Vector3d &tau_c,
    const Eigen::Ref<const Eigen::MatrixXd> &a_pinv,
    const Eigen::Ref<const Eigen::MatrixXd> &null_basis,
    const Eigen::Ref<const Eigen::MatrixXd> &null_proj,
    const Eigen::Ref<const Eigen::VectorXd> &u_0,
    const Eigen::Ref<const Eigen::VectorXd> &max_torque);

class LInfinityAllocator final : public Allocator {
public:
  int kind() const override { return ALLOC_LINFINITY; }
};

} // namespace omni::hardware::controllers

#endif

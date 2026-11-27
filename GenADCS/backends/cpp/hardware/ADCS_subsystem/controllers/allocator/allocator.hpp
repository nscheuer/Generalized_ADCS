#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_ALLOCATOR_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ALLOCATOR_ALLOCATOR_HPP

#include <Eigen/Core>

namespace omni::hardware::controllers {

inline constexpr int ALLOC_L2 = 0;
inline constexpr int ALLOC_LINFINITY = 1;

Eigen::VectorXd saturate_wheels(Eigen::VectorXd u_rw,
                                const Eigen::Ref<const Eigen::VectorXd> &max_torque);

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual int kind() const = 0;
};

} // namespace omni::hardware::controllers

#endif

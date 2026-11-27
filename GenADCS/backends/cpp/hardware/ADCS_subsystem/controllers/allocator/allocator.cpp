#include "allocator.hpp"

#include <utility>

namespace omni::hardware::controllers {

Eigen::VectorXd saturate_wheels(Eigen::VectorXd u_rw,
                                const Eigen::Ref<const Eigen::VectorXd> &max_torque) {
  for (Eigen::Index i = 0; i < u_rw.size(); ++i) {
    const double lim = max_torque[i];
    if (u_rw[i] > lim) {
      u_rw[i] = lim;
    } else if (u_rw[i] < -lim) {
      u_rw[i] = -lim;
    }
  }
  return u_rw;
}

} // namespace omni::hardware::controllers

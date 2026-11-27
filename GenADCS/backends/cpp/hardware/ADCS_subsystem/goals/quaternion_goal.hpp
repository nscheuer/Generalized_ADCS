#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_QUATERNION_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_QUATERNION_GOAL_HPP

#include "goal.hpp"

#include <Eigen/Core>
#include <optional>

namespace omni::hardware::goals {

class QuaternionGoal final : public Goal {
public:
  explicit QuaternionGoal(
      const Eigen::Ref<const Eigen::Vector4d> &quaternion,
      const std::optional<Eigen::Vector3d> &rate = std::nullopt);

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  const Quaternion &quaternion() const { return quaternion_; }
  const Vector3 &rate() const { return rate_; }

private:
  Quaternion quaternion_;
  Vector3 rate_;
};

} // namespace omni::hardware::goals

#endif

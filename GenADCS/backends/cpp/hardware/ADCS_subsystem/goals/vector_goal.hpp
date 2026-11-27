#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_VECTOR_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_VECTOR_GOAL_HPP

#include "goal.hpp"

#include <Eigen/Core>

namespace omni::hardware::goals {

class VectorGoal final : public Goal {
public:
  explicit VectorGoal(
      const Eigen::Ref<const Eigen::Vector3d> &inertial_vector,
      const Eigen::Ref<const Eigen::Vector3d> &body_axis =
          Vector3(0.0, 0.0, 1.0),
      const Eigen::Ref<const Eigen::Vector3d> &inertial_roll =
          Vector3(1.0, 0.0, 0.0),
      const Eigen::Ref<const Eigen::Vector3d> &body_roll =
          Vector3(1.0, 0.0, 0.0));

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  const Vector3 &inertial_vector() const { return inertial_vector_; }
  const Vector3 &body_axis() const { return body_axis_; }
  const Vector3 &inertial_roll() const { return inertial_roll_; }
  const Vector3 &body_roll() const { return body_roll_; }

private:
  Vector3 inertial_vector_;
  Vector3 body_axis_;
  Vector3 inertial_roll_;
  Vector3 body_roll_;
};

} // namespace omni::hardware::goals

#endif

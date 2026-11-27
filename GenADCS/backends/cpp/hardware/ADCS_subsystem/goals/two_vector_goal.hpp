#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_TWO_VECTOR_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_TWO_VECTOR_GOAL_HPP

#include "goal.hpp"

#include <Eigen/Core>

namespace omni::hardware::goals {

class TwoVectorGoal final : public Goal {
public:
  TwoVectorGoal(const Eigen::Matrix<double, 2, 3> &inertial_vectors,
                const Eigen::Matrix<double, 2, 3> &body_vectors);

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  const Vector3 &inertial_first() const { return inertial_first_; }
  const Vector3 &inertial_second() const { return inertial_second_; }
  const Vector3 &body_first() const { return body_first_; }
  const Vector3 &body_second() const { return body_second_; }

private:
  Vector3 inertial_first_;
  Vector3 inertial_second_;
  Vector3 body_first_;
  Vector3 body_second_;
};

} // namespace omni::hardware::goals

#endif

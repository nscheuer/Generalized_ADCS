#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_GOAL_HPP

#include <Eigen/Core>
#include <optional>
#include <string>
#include <utility>

namespace omni::environment {
class DE421;
} // namespace omni::environment

namespace omni::hardware::goals {

using Quaternion = Eigen::Vector4d;
using Vector3 = Eigen::Vector3d;
using GoalReference = std::pair<std::optional<Quaternion>, std::optional<Vector3>>;

struct GoalRuntime {
  double mu = 3.986004418e14;
  const omni::environment::DE421 *ephemeris = nullptr;
};

class Goal {
public:
  virtual ~Goal() = default;

  virtual GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                                  double epoch_jd,
                                  const GoalRuntime &runtime = {}) const = 0;

  GoalReference operator()(const Eigen::Ref<const Eigen::VectorXd> &state,
                           double epoch_jd,
                           const GoalRuntime &runtime = {}) const {
    return reference(state, epoch_jd, runtime);
  }
};

void validate_reference_state(const Eigen::Ref<const Eigen::VectorXd> &state);
Quaternion validate_quaternion(const Eigen::Ref<const Eigen::Vector4d> &quaternion);
Vector3 validate_vector(const Eigen::Ref<const Eigen::Vector3d> &vector,
                        const std::string &name = "vector");

namespace detail {
Quaternion triad_quaternion(const Vector3 &i0, const Vector3 &i_roll,
                            const Vector3 &b0, const Vector3 &b_roll);
} // namespace detail

} // namespace omni::hardware::goals

#endif

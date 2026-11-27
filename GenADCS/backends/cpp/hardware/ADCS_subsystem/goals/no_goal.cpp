#include "no_goal.hpp"

namespace omni::hardware::goals {

GoalReference NoGoal::reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                                double /*epoch_jd*/,
                                const GoalRuntime & /*runtime*/) const {
  validate_reference_state(state);
  return {std::nullopt, std::nullopt};
}

} // namespace omni::hardware::goals

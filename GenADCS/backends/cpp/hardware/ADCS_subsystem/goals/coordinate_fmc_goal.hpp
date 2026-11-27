#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_COORDINATE_FMC_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_COORDINATE_FMC_GOAL_HPP

#include "coordinate_goal.hpp"

namespace omni::hardware::goals {

class CoordinateFMCGoal final : public CoordinateGoal {
public:
  CoordinateFMCGoal(double lat, double lon, double alt = 0.0,
                    int fmc_level = 1);

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  int fmc_level() const { return fmc_level_; }

private:
  int fmc_level_;
};

} // namespace omni::hardware::goals

#endif

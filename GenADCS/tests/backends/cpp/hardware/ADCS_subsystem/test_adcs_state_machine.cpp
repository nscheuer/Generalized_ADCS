#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/bdot.hpp"
#include "hardware/ADCS_subsystem/goals/nadir_goal.hpp"
#include "hardware/ADCS_subsystem/goals/quaternion_goal.hpp"
#include "hardware/ADCS_subsystem/state_machine.hpp"

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace omni::hardware;
using namespace omni::hardware::goals;
using namespace omni::hardware::controllers;

int main() {
  GoalTable gt;
  ControllerTable ct;

  auto q_goal = std::make_shared<QuaternionGoal>(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
  auto bdot = std::make_shared<BDot>(5e4);
  ADCSMode detumble("detumble", std::make_shared<NoGoal>(), bdot, {},
                    {{"Magnetorquer", true}});
  ADCSMode point("point", std::make_shared<NadirGoal>(),
                 std::make_shared<NoController>(), {{"Gyroscope", false}}, {});

  ADCSStateMachine sm("detumble", {detumble, point},
                      {{1.5, std::string("point")}, {3.0, q_goal}});
  const ADCSSMRuntime rt =
      prepare_adcs_sm(&sm, &gt, &ct, {"Gyroscope"}, {}, {"Magnetorquer"});
  CHECK(rt.enabled);
  CHECK(rt.initial_mode == 0);
  CHECK(rt.mode_names[0] == "detumble");
  CHECK(rt.mode_names[1] == "point");
  CHECK(rt.mode_sensor_mask[1][0] == 0);
  CHECK(rt.mode_mtq_mask[0][0] == 1);

  auto work = init_adcs_work(rt);
  CHECK(work[0] == 0);
  CHECK(work[1] == 0);

  // Commanded mode switch from OBC overrides mode and resets to mode's default goal.
  auto [m1, g1] = step_adcs_sm(0.5, rt, work, 1, -1);
  CHECK(m1 == 1);
  CHECK(rt.goal_names[static_cast<std::size_t>(g1)] == "NadirGoal");

  // Standalone timeline event at t=3.0 overrides goal to QuaternionGoal.
  auto [m2, g2] = step_adcs_sm(3.0, rt, work, -1, -1);
  CHECK(m2 == 1);
  CHECK(rt.goal_names[static_cast<std::size_t>(g2)] == "QuaternionGoal");

  Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
  state[0] = 1.0;
  state[7] = 6878137.0;
  state[11] = 7612.6;
  const auto [q_ref, w_ref] = evaluate_goal(g2, rt, state, 2461000.5);
  CHECK(q_ref.isApprox(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), 1e-12));
  CHECK(w_ref.isApprox(Eigen::Vector3d::Zero(), 1e-12));

  // Invalid unknown timeline mode throws.
  bool threw = false;
  try {
    ADCSStateMachine bad("detumble", {detumble},
                         {{1.0, std::string("missing")}});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
  return 0;
}

#include "check.hpp"
#include "hardware/ADCS_subsystem/goals/nadir_goal.hpp"
#include "hardware/ADCS_subsystem/goals/quaternion_goal.hpp"
#include "hardware/OBC_subsystem/runtime.hpp"
#include "helpers/math/frames/conversions.hpp"

#include <Eigen/Core>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omni::hardware::obc;
using namespace omni::hardware::goals;

namespace {

Eigen::VectorXd make_state_vec(
    const Eigen::Vector3d &rates = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d &pos = Eigen::Vector3d(6878137.0, 0.0, 0.0),
    const Eigen::Vector3d &vel = Eigen::Vector3d(0.0, 7612.6, 0.0)) {
  Eigen::VectorXd y = Eigen::VectorXd::Zero(13);
  y[0] = 1.0;
  y.segment<3>(4) = rates;
  y.segment<3>(7) = pos;
  y.segment<3>(10) = vel;
  return y;
}

void test_obc_telecommand_same_step_preserves_adcs_binding_and_goal() {
  GoalTable gt;
  const std::map<std::string, int> adcs_modes = {{"detumble", 0},
                                                 {"nominal", 1}};
  auto q_override = std::make_shared<QuaternionGoal>(
      Eigen::Vector4d(0.0, 1.0, 0.0, 0.0));

  ModeBindings boot_b;
  boot_b.adcs = std::string("detumble");
  ModeBindings nom_b;
  nom_b.adcs = std::make_pair(std::string("nominal"),
                              std::make_shared<NadirGoal>());

  auto sm = std::make_shared<StateMachine>(
      "boot",
      std::vector<Mode>{
          Mode("boot", {}, std::nullopt, std::nullopt, std::nullopt, boot_b),
          Mode("nominal", {}, std::nullopt, std::nullopt, std::nullopt, nom_b),
          Mode("logging"),
      },
      std::vector<Transition>{
          Transition("boot", "nominal",
                     std::make_shared<Telecommand>("GO_NOMINAL", q_override)),
          Transition("nominal", "logging",
                     std::make_shared<Telecommand>("START_LOG")),
      },
      std::vector<TimelineEntry>{
          TimelineEntry{1.0, "GO_NOMINAL", nullptr},
          TimelineEntry{1.0, "START_LOG", nullptr},
      });
  OBCSubsystem obc(sm);
  const OBCRuntime rt = prepare_obc(&obc, adcs_modes, &gt);
  auto work = init_obc_work(rt);
  const Eigen::VectorXd y = make_state_vec();

  const auto [state, cm, cg] = step_obc(y, 1.0, 2461000.5, 0.5, rt, work);
  CHECK(rt.mode_names[static_cast<std::size_t>(state)] == "logging");
  CHECK(cm == 1);
  CHECK(gt.names[static_cast<std::size_t>(cg)] == "QuaternionGoal");
}

void test_obc_after_and_completion_transitions() {
  GoalTable gt;
  const std::map<std::string, int> adcs_modes = {{"idle", 0}};
  auto sm = std::make_shared<StateMachine>(
      "warmup",
      std::vector<Mode>{
          Mode("warmup"),
          Mode("work", {}, std::nullopt, 3.0),
          Mode("done"),
      },
      std::vector<Transition>{
          Transition("warmup", "work", std::make_shared<After>(2.0)),
          Transition("work", "done", std::make_shared<Completion>()),
      });
  OBCSubsystem obc(sm);
  const OBCRuntime rt = prepare_obc(&obc, adcs_modes, &gt);
  auto work = init_obc_work(rt);
  const Eigen::VectorXd y = make_state_vec();

  const auto [s0, cm0, cg0] = step_obc(y, 0.0, 2461000.5, 0.5, rt, work);
  (void)cm0;
  (void)cg0;
  CHECK(rt.mode_names[static_cast<std::size_t>(s0)] == "warmup");

  const auto [s1, cm1, cg1] = step_obc(y, 2.0, 2461000.5, 0.5, rt, work);
  (void)cm1;
  (void)cg1;
  CHECK(rt.mode_names[static_cast<std::size_t>(s1)] == "work");

  const auto [s2, cm2, cg2] = step_obc(y, 4.9, 2461000.5, 0.5, rt, work);
  (void)cm2;
  (void)cg2;
  CHECK(rt.mode_names[static_cast<std::size_t>(s2)] == "work");

  const auto [s3, cm3, cg3] = step_obc(y, 5.0, 2461000.5, 0.5, rt, work);
  (void)cm3;
  (void)cg3;
  CHECK(rt.mode_names[static_cast<std::size_t>(s3)] == "done");
}

void test_obc_infov_entry_and_exit() {
  const double jd = 2461000.5;
  const Eigen::Vector3d r_eci(6878137.0, 0.0, 0.0);
  const Eigen::Vector3d r_ecef =
      omni::frames::ecef_to_gcrf_matrix(jd, jd).transpose() * r_eci;
  const double pi = std::acos(-1.0);
  const double subsat_lon = std::atan2(r_ecef[1], r_ecef[0]) * 180.0 / pi;
  auto swath = std::make_shared<InFOV>(0.0, subsat_lon, 0.0, 15.0);
  auto swath_exit = std::make_shared<InFOV>(swath->inverted());

  GoalTable gt;
  const std::map<std::string, int> adcs_modes = {{"idle", 0}};
  auto sm = std::make_shared<StateMachine>(
      "idle", std::vector<Mode>{Mode("idle"), Mode("payload")},
      std::vector<Transition>{
          Transition("idle", "payload", swath),
          Transition("payload", "idle", swath_exit),
      });
  OBCSubsystem obc(sm);
  const OBCRuntime rt = prepare_obc(&obc, adcs_modes, &gt);
  auto work = init_obc_work(rt);

  const Eigen::VectorXd y_over =
      make_state_vec(Eigen::Vector3d::Zero(), Eigen::Vector3d(6878137.0, 0.0, 0.0));
  const auto [s_in, cm_in, cg_in] = step_obc(y_over, 0.0, jd, 0.5, rt, work);
  (void)cm_in;
  CHECK(rt.mode_names[static_cast<std::size_t>(s_in)] == "payload");
  CHECK(gt.names[static_cast<std::size_t>(cg_in)] == "CoordinateGoal");

  const Eigen::VectorXd y_far = make_state_vec(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(-6878137.0, 0.0, 0.0));
  const auto [s_out, cm_out, cg_out] = step_obc(y_far, 1.0, jd, 0.5, rt, work);
  (void)cm_out;
  (void)cg_out;
  CHECK(rt.mode_names[static_cast<std::size_t>(s_out)] == "idle");
}

void test_obc_fdir_triggered_custom_target_and_unlatch() {
  GoalTable gt;
  const std::map<std::string, int> adcs_modes = {{"nominal", 0}, {"safe", 1}};

  ModeBindings nom_b;
  nom_b.adcs = std::string("nominal");
  ModeBindings safe_b;
  safe_b.adcs = std::string("safe");

  auto sm = std::make_shared<StateMachine>(
      "nominal",
      std::vector<Mode>{
          Mode("nominal", {}, std::nullopt, std::nullopt, std::nullopt, nom_b),
          Mode("alt_safe", {}, std::nullopt, std::nullopt, std::nullopt,
               safe_b),
          Mode("safe_mode", {}, std::nullopt, std::nullopt, std::nullopt,
               safe_b),
      },
      std::vector<Transition>{
          Transition("nominal", "alt_safe",
                     std::make_shared<FDIRTriggered>("rate_mon")),
          Transition("alt_safe", "nominal", std::make_shared<After>(2.0)),
      });
  auto fdir = std::make_shared<FDIR>(
      std::vector<std::shared_ptr<FDIRMonitor>>{
          std::make_shared<AngularRateMonitor>(0.05, "rate_mon")});
  OBCSubsystem obc(sm, fdir);
  const OBCRuntime rt = prepare_obc(&obc, adcs_modes, &gt);
  auto work = init_obc_work(rt);

  const Eigen::VectorXd y_fast =
      make_state_vec(Eigen::Vector3d(0.1, 0.0, 0.0));
  const auto [s1, cm1, cg1] = step_obc(y_fast, 0.0, 2461000.5, 0.5, rt, work);
  (void)cg1;
  CHECK(rt.mode_names[static_cast<std::size_t>(s1)] == "alt_safe");
  CHECK(cm1 == 1);
  CHECK(work[3] == 1.0);

  const Eigen::VectorXd y_calm = make_state_vec();
  const auto [s2, cm2, cg2] = step_obc(y_calm, 2.0, 2461000.5, 0.5, rt, work);
  (void)cg2;
  CHECK(rt.mode_names[static_cast<std::size_t>(s2)] == "nominal");
  CHECK(cm2 == 0);
  CHECK(work[3] == 0.0);

  const auto [s3, cm3, cg3] = step_obc(y_fast, 2.5, 2461000.5, 0.5, rt, work);
  (void)cm3;
  (void)cg3;
  CHECK(rt.mode_names[static_cast<std::size_t>(s3)] == "alt_safe");
  CHECK(work[3] == 1.0);
}

void test_obc_fdir_triggered_unknown_monitor_raises() {
  auto sm = std::make_shared<StateMachine>(
      "nominal",
      std::vector<Mode>{Mode("nominal"), Mode("safe_mode")},
      std::vector<Transition>{
          Transition("nominal", "safe_mode",
                     std::make_shared<FDIRTriggered>("missing"))});
  auto fdir = std::make_shared<FDIR>(
      std::vector<std::shared_ptr<FDIRMonitor>>{
          std::make_shared<AngularRateMonitor>(0.05, "rate_mon")});
  OBCSubsystem obc(sm, fdir);
  bool threw = false;
  try {
    (void)prepare_obc(&obc);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

} // namespace

int main() {
  test_obc_telecommand_same_step_preserves_adcs_binding_and_goal();
  test_obc_after_and_completion_transitions();
  test_obc_infov_entry_and_exit();
  test_obc_fdir_triggered_custom_target_and_unlatch();
  test_obc_fdir_triggered_unknown_monitor_raises();
  return 0;
}

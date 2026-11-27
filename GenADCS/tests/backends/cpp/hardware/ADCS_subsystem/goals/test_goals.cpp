#include "check.hpp"
#include "environment/ephemeris/de421/de421.hpp"
#include "hardware/ADCS_subsystem/goals/goals.hpp"
#include "helpers/math/frames/conversions.hpp"
#include "helpers/math/rotations/rotation_types.hpp"
#include "helpers/math/time/conversions.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

namespace goals = omni::hardware::goals;
namespace rot = omni::rotations;

template <typename Fn> void expect_invalid_argument(Fn &&fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

Eigen::VectorXd circular_leo_state() {
  Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
  state[0] = 1.0; // identity quaternion
  const double r = 6878137.0;
  const double mu = 3.986004418e14;
  const double v = std::sqrt(mu / r);
  state[7] = r;
  state[11] = v;
  return state;
}

void test_validation_helpers() {
  Eigen::VectorXd short_state = Eigen::VectorXd::Zero(12);
  short_state[0] = 1.0;
  expect_invalid_argument(
      [&]() { goals::validate_reference_state(short_state); });

  Eigen::VectorXd zero_q_state = Eigen::VectorXd::Zero(13);
  expect_invalid_argument(
      [&]() { goals::validate_reference_state(zero_q_state); });

  Eigen::VectorXd nan_state = circular_leo_state();
  nan_state[8] = std::numeric_limits<double>::quiet_NaN();
  expect_invalid_argument(
      [&]() { goals::validate_reference_state(nan_state); });

  const Eigen::Vector4d q_norm =
      goals::validate_quaternion(Eigen::Vector4d(2.0, 0.0, 0.0, 0.0));
  CHECK(std::abs(q_norm[0] - 1.0) < 1e-12);
  CHECK(std::abs(q_norm.norm() - 1.0) < 1e-12);

  expect_invalid_argument(
      []() { goals::validate_quaternion(Eigen::Vector4d::Zero()); });
  expect_invalid_argument([]() {
    goals::validate_quaternion(
        Eigen::Vector4d(1.0, 0.0, std::numeric_limits<double>::infinity(), 0.0));
  });

  const Eigen::Vector3d v_norm =
      goals::validate_vector(Eigen::Vector3d(0.0, 3.0, 4.0), "test_vec");
  CHECK(std::abs(v_norm.norm() - 1.0) < 1e-12);
  CHECK(std::abs(v_norm[1] - 0.6) < 1e-12);
  CHECK(std::abs(v_norm[2] - 0.8) < 1e-12);

  expect_invalid_argument(
      []() { goals::validate_vector(Eigen::Vector3d::Zero()); });
  expect_invalid_argument([]() {
    goals::validate_vector(
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0));
  });
}

void test_no_goal() {
  const Eigen::VectorXd state = circular_leo_state();
  goals::NoGoal goal;
  const auto [q, w] = goal.reference(state, 2460000.5);
  CHECK(!q.has_value());
  CHECK(!w.has_value());

  const auto [q2, w2] = goal(state, 2460000.5);
  CHECK(!q2.has_value());
  CHECK(!w2.has_value());
}

void test_quaternion_goal() {
  const Eigen::VectorXd state = circular_leo_state();
  goals::QuaternionGoal goal(Eigen::Vector4d(0.0, 2.0, 0.0, 0.0));
  const auto [q, w] = goal.reference(state, 2460000.5);
  CHECK(q.has_value());
  CHECK(w.has_value());
  CHECK(std::abs(q->norm() - 1.0) < 1e-12);
  CHECK(std::abs((*q)[1] - 1.0) < 1e-12);
  CHECK(w->norm() < 1e-12);

  const Eigen::Vector3d custom_rate(0.01, -0.02, 0.03);
  goals::QuaternionGoal goal_with_rate(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                                       custom_rate);
  const auto [q2, w2] = goal_with_rate(state, 2460000.5);
  CHECK(q2.has_value());
  CHECK(w2.has_value());
  CHECK((*w2 - custom_rate).norm() < 1e-12);

  expect_invalid_argument([]() {
    goals::QuaternionGoal(
        Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, std::numeric_limits<double>::infinity(), 0.0));
  });
}

void test_angular_rate_goal() {
  const Eigen::VectorXd state = circular_leo_state();
  const Eigen::Vector3d rate(0.1, -0.2, 0.3);
  goals::AngularRateGoal goal(rate);
  const auto [q, w] = goal.reference(state, 2460000.5);
  CHECK(!q.has_value());
  CHECK(w.has_value());
  CHECK((*w - rate).norm() < 1e-12);

  expect_invalid_argument([]() {
    goals::AngularRateGoal(
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
  });
}

void test_vector_and_two_vector_goals() {
  const Eigen::VectorXd state = circular_leo_state();

  const Eigen::Vector3d inertial_vec(0.0, 1.0, 0.0);
  const Eigen::Vector3d body_axis(0.0, 0.0, 1.0);
  goals::VectorGoal vgoal(inertial_vec, body_axis,
                          Eigen::Vector3d(1.0, 0.0, 0.0),
                          Eigen::Vector3d(1.0, 0.0, 0.0));
  const auto [q, w] = vgoal.reference(state, 2460000.5);
  CHECK(q.has_value());
  CHECK(w.has_value());
  CHECK(std::abs(q->norm() - 1.0) < 1e-12);
  CHECK(w->norm() < 1e-12);
  const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
  CHECK((dcm * inertial_vec - body_axis).norm() < 1e-12);

  expect_invalid_argument([]() {
    goals::VectorGoal(Eigen::Vector3d(1.0, 0.0, 0.0),
                      Eigen::Vector3d(0.0, 0.0, 1.0),
                      Eigen::Vector3d(-2.0, 0.0, 0.0),
                      Eigen::Vector3d(1.0, 0.0, 0.0));
  });
  expect_invalid_argument([]() {
    goals::VectorGoal(Eigen::Vector3d(0.0, 1.0, 0.0),
                      Eigen::Vector3d(0.0, 0.0, 1.0),
                      Eigen::Vector3d(1.0, 0.0, 0.0),
                      Eigen::Vector3d(0.0, 0.0, 5.0));
  });

  Eigen::Matrix<double, 2, 3> inertial_pair;
  inertial_pair << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
  Eigen::Matrix<double, 2, 3> body_pair;
  body_pair << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  goals::TwoVectorGoal tvgoal(inertial_pair, body_pair);
  const auto [q2, w2] = tvgoal.reference(state, 2460000.5);
  CHECK(q2.has_value());
  CHECK(w2.has_value());
  CHECK(std::abs(q2->norm() - 1.0) < 1e-12);
  CHECK(w2->norm() < 1e-12);
  const Eigen::Matrix3d dcm2 = rot::quaternion_to_dcm(*q2);
  CHECK((dcm2 * inertial_pair.row(0).transpose() - body_pair.row(0).transpose())
            .norm() < 1e-12);
  CHECK((dcm2 * inertial_pair.row(1).transpose() - body_pair.row(1).transpose())
            .norm() < 1e-12);

  Eigen::Matrix<double, 2, 3> parallel_pair;
  parallel_pair << 1.0, 0.0, 0.0, -1.0, 0.0, 0.0;
  expect_invalid_argument(
      [&]() { goals::TwoVectorGoal(parallel_pair, body_pair); });
}

void test_orbit_goals() {
  const Eigen::VectorXd state = circular_leo_state();
  const double epoch_jd = 2460000.5;
  const goals::GoalRuntime runtime{};
  const Eigen::Vector3d r_hat = state.segment<3>(7).normalized();
  const Eigen::Vector3d v_hat = state.segment<3>(10).normalized();
  const Eigen::Vector3d n_hat =
      state.segment<3>(7).cross(state.segment<3>(10)).normalized();

  {
    goals::NadirGoal nadir;
    const auto [q, w] = nadir.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    CHECK(std::abs(q->norm() - 1.0) < 1e-12);
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() + r_hat).norm() < 1e-12);
    CHECK((dcm.row(0).transpose() - v_hat).norm() < 1e-12);
    const double r = state.segment<3>(7).norm();
    const double omega_orbit = std::sqrt(runtime.mu / (r * r * r));
    CHECK(std::abs(w->norm() - omega_orbit) < 1e-9);
    // Body +Y is (-radial) x tangent = +orbit_normal, so orbit rotation is +Y.
    CHECK(std::abs((*w)[1] - omega_orbit) < 1e-9);
  }
  {
    goals::ZenithGoal zenith;
    const auto [q, w] = zenith.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - r_hat).norm() < 1e-12);
  }
  {
    goals::AlongTrackGoal along;
    const auto [q, w] = along.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - v_hat).norm() < 1e-12);
  }
  {
    goals::CrossTrackGoal cross;
    const auto [q, w] = cross.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - n_hat).norm() < 1e-12);
  }
  {
    goals::VelocityGoal vel;
    const auto [q, w] = vel.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - v_hat).norm() < 1e-12);
  }
  {
    goals::SunGoal sun;
    const auto [q, w] = sun.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    CHECK(std::abs(q->norm() - 1.0) < 1e-12);
    CHECK(w->allFinite());
    const omni::environment::DE421 de421;
    const double tdb = omni::time::utc_jd_to_tdb_jd(epoch_jd);
    const Eigen::Vector3d sun_pos = de421.position(0, tdb) * 1000.0;
    const Eigen::Vector3d expected_z =
        (sun_pos - state.segment<3>(7)).normalized();
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - expected_z).norm() < 1e-12);
  }

  expect_invalid_argument([&]() {
    goals::NadirGoal().reference(
        state, std::numeric_limits<double>::quiet_NaN(), runtime);
  });
  Eigen::VectorXd zero_pos = state;
  zero_pos.segment<3>(7).setZero();
  expect_invalid_argument(
      [&]() { goals::NadirGoal().reference(zero_pos, epoch_jd, runtime); });
}

void test_coordinate_goals() {
  const Eigen::VectorXd state = circular_leo_state();
  const double epoch_jd = 2460000.5;
  const goals::GoalRuntime runtime{};

  const Eigen::Vector3d ecef_eq =
      goals::CoordinateGoal::geodetic_to_ecef(0.0, 0.0, 0.0);
  CHECK((ecef_eq - Eigen::Vector3d(6378137.0, 0.0, 0.0)).norm() < 1e-6);

  expect_invalid_argument([]() { goals::CoordinateGoal(91.0, 0.0); });
  expect_invalid_argument([]() { goals::CoordinateGoal(0.0, -181.0); });
  expect_invalid_argument([]() { goals::CoordinateGoal(0.0, 0.0, -6400.0); });
  expect_invalid_argument([]() {
    goals::CoordinateGoal(std::numeric_limits<double>::quiet_NaN(), 0.0);
  });

  {
    goals::CoordinateGoal coord(10.0, 20.0, 0.5);
    const auto [q, w] = coord.reference(state, epoch_jd, runtime);
    CHECK(q.has_value() && w.has_value());
    CHECK(std::abs(q->norm() - 1.0) < 1e-12);
    CHECK(w->allFinite() && w->norm() > 0.0);
    const Eigen::Vector3d target_gcrf =
        omni::frames::ecef_to_gcrf_matrix(epoch_jd, epoch_jd) *
        coord.target_ecef_m();
    const Eigen::Vector3d expected_z =
        (target_gcrf - state.segment<3>(7)).normalized();
    const Eigen::Matrix3d dcm = rot::quaternion_to_dcm(*q);
    CHECK((dcm.row(2).transpose() - expected_z).norm() < 1e-12);
  }

  expect_invalid_argument([]() { goals::CoordinateFMCGoal(0.0, 0.0, 0.0, 0); });
  expect_invalid_argument(
      []() { goals::CoordinateFMCGoal(0.0, 0.0, 0.0, 11); });

  {
    goals::CoordinateFMCGoal fmc1(0.0, 45.0, 0.0, 1);
    const auto [q1, w1] = fmc1.reference(state, epoch_jd, runtime);
    CHECK(q1.has_value() && w1.has_value());
    const Eigen::Matrix3d dcm1 = rot::quaternion_to_dcm(*q1);
    const Eigen::Vector3d nadir = -state.segment<3>(7).normalized();
    CHECK((dcm1.row(2).transpose() - nadir).norm() < 1e-12);

    goals::CoordinateFMCGoal fmc2(0.0, 45.0, 0.0, 2);
    const auto [q2, w2] = fmc2.reference(state, epoch_jd, runtime);
    CHECK(q2.has_value() && w2.has_value());
    CHECK(std::abs(q2->norm() - 1.0) < 1e-12);
    CHECK(w2->allFinite() && w2->norm() > 0.0);
    const Eigen::Matrix3d dcm2 = rot::quaternion_to_dcm(*q2);
    CHECK(std::abs(dcm2.row(2).transpose().dot(nadir) - 0.5) < 1e-12);
  }
}

} // namespace

int main() {
  test_validation_helpers();
  test_no_goal();
  test_quaternion_goal();
  test_angular_rate_goal();
  test_vector_and_two_vector_goals();
  test_orbit_goals();
  test_coordinate_goals();
  std::cout << "All C++ ADCS goals tests passed.\n";
  return 0;
}

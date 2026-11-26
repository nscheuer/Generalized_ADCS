#include "separate_solver.hpp"

#include "../orbital_propagators/keplerian_orbit_propagator.hpp"
#include "reference_attitude_propagator.hpp"
#include "rk4_attitude_propagator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::propagation {
SeparateTrajectory integrate_separate_reference(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const std::string &mode,
    const SeparateProgress &progress) {
  auto trajectory = KeplerianOrbitPropagator(dt).integrate(
      initial, mu, tf, record_every, progress);
  ReferenceAttitudePropagator(dt,
      ReferenceAttitudePropagator::parse_mode(mode))
      .apply(trajectory.second, trajectory.first, initial_julian_date, mu);
  return trajectory;
}

SeparateTrajectory integrate_separate_rk4(
    const Eigen::VectorXd &initial, double mu, double initial_julian_date,
    double dt, double tf, int record_every, const Runtime &runtime,
    const SeparateProgress &progress) {
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
  if (!std::isfinite(tf) || tf < 0.0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  if (record_every < 1) {
    throw std::invalid_argument("record_every must be a positive integer");
  }
  if (initial.size() != 13 + runtime.wheel_commands.size()) {
    throw std::invalid_argument("state has incompatible wheel momentum size");
  }
  Vector state = initial;
  const double norm = state.head<4>().norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    throw std::invalid_argument("attitude quaternion must have nonzero norm");
  }
  state.head<4>() /= norm;
  if (runtime.wheel_commands.size() != 0) {
    project(state, runtime);
  }

  const auto full_steps = static_cast<std::size_t>(std::floor(tf / dt));
  double remainder = tf - static_cast<double>(full_steps) * dt;
  if (remainder <= 1e-12 * std::max(1.0, tf)) {
    remainder = 0.0;
  }
  const std::size_t total_steps = full_steps + (remainder > 0.0 ? 1 : 0);
  const auto stride = static_cast<std::size_t>(record_every);
  const std::size_t sample_count =
      1 + total_steps / stride + (total_steps % stride != 0 ? 1 : 0);
  SeparateTrajectory trajectory;
  trajectory.first.reserve(sample_count);
  trajectory.second.resize(static_cast<Eigen::Index>(sample_count), state.size());
  trajectory.first.push_back(0.0);
  trajectory.second.row(0) = state.transpose();

  for (std::size_t step = 0; step < total_steps; ++step) {
    const double step_dt = step == full_steps ? remainder : dt;
    // Match the Python split step: attitude uses this step's starting orbit,
    // then Keplerian propagation advances the orbit for the next step.
    state = rk4_attitude_step(
        state, initial_julian_date + static_cast<double>(step) * dt / 86400.0,
        step_dt, runtime);
    Eigen::Vector3d next_velocity;
    const Eigen::Vector3d next_position = keplerian_position_velocity_step(
        state.segment<3>(7), state.segment<3>(10), mu, step_dt, next_velocity);
    state.segment<3>(7) = next_position;
    state.segment<3>(10) = next_velocity;

    const std::size_t completed = step + 1;
    if (completed % stride == 0 || completed == total_steps) {
      const auto row = static_cast<Eigen::Index>(trajectory.first.size());
      trajectory.second.row(row) = state.transpose();
      trajectory.first.push_back(completed == total_steps
          ? tf : std::min(static_cast<double>(completed) * dt, tf));
    }
    if (progress) {
      progress(1);
    }
  }
  return trajectory;
}
} // namespace omni::propagation

#include "rk4_attitude_propagator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::propagation {
namespace {
constexpr double seconds_per_day = 86400.0;

Vector attitude_derivative(const Vector &state, double jd,
                           const Runtime &runtime) {
  Vector result = derivative(state, jd, runtime);
  // Orbit position and velocity are frozen during attitude-only propagation.
  result.segment<6>(7).setZero();
  return result;
}

Vector normalized(Vector state, const Runtime &runtime) {
  const double norm = state.segment<4>(0).norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    throw std::invalid_argument("attitude quaternion must have nonzero norm");
  }
  state.segment<4>(0) /= norm;
  if (runtime.wheel_commands.size() != 0) {
    project(state, runtime);
  }
  return state;
}
} // namespace

Vector rk4_attitude_step(const Vector &input, double jd, double dt,
                         const Runtime &runtime) {
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
  if (input.size() != 13 + runtime.wheel_commands.size()) {
    throw std::invalid_argument("state has incompatible wheel momentum size");
  }
  const Vector state = normalized(input, runtime);
  const Vector k1 = attitude_derivative(state, jd, runtime);
  const Vector k2 = attitude_derivative(state + 0.5 * dt * k1,
                                        jd + 0.5 * dt / seconds_per_day, runtime);
  const Vector k3 = attitude_derivative(state + 0.5 * dt * k2,
                                        jd + 0.5 * dt / seconds_per_day, runtime);
  const Vector k4 = attitude_derivative(state + dt * k3,
                                        jd + dt / seconds_per_day, runtime);
  Vector result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  return normalized(std::move(result), runtime);
}

Trajectory integrate_attitude_rk4(Vector state, double jd, double dt,
                                  double tf, const Runtime &runtime,
                                  std::size_t record_every) {
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
  if (!std::isfinite(tf) || tf < 0.0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  if (record_every == 0) {
    throw std::invalid_argument("record_every must be a positive integer");
  }
  state = normalized(std::move(state), runtime);
  const auto full_steps = static_cast<std::size_t>(std::floor(tf / dt));
  double remainder = tf - static_cast<double>(full_steps) * dt;
  if (remainder <= 1e-12 * std::max(1.0, tf)) {
    remainder = 0.0;
  }
  const std::size_t total_steps = full_steps + (remainder > 0.0 ? 1 : 0);
  Trajectory output;
  output.times.push_back(0.0);
  output.states.push_back(state);
  for (std::size_t row = 0; row < total_steps; ++row) {
    const double step_dt = row == full_steps ? remainder : dt;
    state = rk4_attitude_step(state, jd + static_cast<double>(row) * dt /
                                          seconds_per_day,
                              step_dt, runtime);
    const std::size_t completed = row + 1;
    if (completed % record_every == 0 || completed == total_steps) {
      output.times.push_back(std::min(static_cast<double>(completed) * dt, tf));
      output.states.push_back(state);
    }
  }
  if (output.times.back() != tf) {
    output.times.push_back(tf);
    output.states.push_back(state);
  }
  return output;
}
} // namespace omni::propagation

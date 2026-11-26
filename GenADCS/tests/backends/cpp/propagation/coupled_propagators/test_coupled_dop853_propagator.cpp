#include "propagation/coupled_propagators/coupled_dop853_propagator.hpp"
#include "propagation/coupled_propagators/integrate.hpp"

#include <cmath>
#include <stdexcept>

int main() {
  using namespace omni::propagation;
  Runtime runtime;
  runtime.mu = 3.986004418e14;
  runtime.radius = 6378137.0;
  runtime.mass = 10.0;
  runtime.wheel_commands.resize(0);

  Vector state = Vector::Zero(13);
  state[0] = 1.0;
  state[6] = 0.1;
  state[7] = 7.0e6;
  state[11] = 7500.0;

  const Vector deriv = derivative(state, 2451545.0, runtime);
  const Vector atol = Vector::Constant(13, 1e-12);
  const auto [next, f_next, err] =
      dop853_attempt(state, deriv, 2451545.0, 0.5, runtime, 1e-9, atol);
  (void)f_next;
  if (!std::isfinite(err) || err < 0.0 ||
      std::abs(next[0] - std::cos(0.025)) > 1e-10 || next[8] <= 0.0) {
    throw std::runtime_error("coupled DOP853 attempt mismatch");
  }

  const auto trajectory = integrate_dop853(
      state, 2451545.0, 0.5, 1.0, runtime, 1e-9, atol, 10.0, 1e-6, {}, 500);
  if (trajectory.times.empty() ||
      std::abs(trajectory.times.back() - 1.0) > 1e-12 ||
      std::abs(trajectory.states.back()[0] - std::cos(0.05)) > 1e-10 ||
      trajectory.states.back()[8] <= 0.0) {
    throw std::runtime_error("coupled DOP853 integration mismatch");
  }

  return 0;
}

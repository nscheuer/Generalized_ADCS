#include "propagation/coupled_propagators/coupled_rk4_propagator.hpp"
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

  const Vector stepped = rk4_step(state, 2451545.0, 1.0, runtime);
  const double angle = 0.05;
  if (std::abs(stepped[0] - std::cos(angle)) > 1e-8 ||
      std::abs(stepped[3] + std::sin(angle)) > 1e-8 ||
      stepped[8] <= 0.0) {
    throw std::runtime_error("coupled RK4 step mismatch");
  }

  const auto trajectory =
      integrate_rk4(state, 2451545.0, 0.25, 1.1, runtime, {}, 3);
  if (trajectory.times.size() != 3 ||
      std::abs(trajectory.times[1] - 0.75) > 1e-12 ||
      std::abs(trajectory.times.back() - 1.1) > 1e-12 ||
      std::abs(trajectory.states.back()[0] - std::cos(0.055)) > 1e-9 ||
      std::abs(trajectory.states.back()[3] + std::sin(0.055)) > 1e-9 ||
      trajectory.states.back()[8] <= 0.0) {
    throw std::runtime_error("coupled RK4 integration mismatch");
  }

  return 0;
}

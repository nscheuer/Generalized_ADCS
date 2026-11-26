#pragma once

#include "../coupled_propagators/_coupled_dynamics.hpp"
#include "../coupled_propagators/integrate.hpp"

namespace omni::propagation {

Vector rk4_attitude_step(const Vector &state, double julian_date,
                         double step_seconds, const Runtime &runtime);
Trajectory integrate_attitude_rk4(Vector state, double julian_date,
                                  double step_seconds, double final_seconds,
                                  const Runtime &runtime,
                                  std::size_t record_every = 1);

} // namespace omni::propagation

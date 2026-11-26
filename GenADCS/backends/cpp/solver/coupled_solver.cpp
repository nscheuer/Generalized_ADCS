#include "coupled_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::solver::detail {

void validate_coupled_solver_options(double final_time, int chunk_steps) {
  if (!std::isfinite(final_time) || final_time < 0.0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  if (chunk_steps <= 0) {
    throw std::invalid_argument(
        "progress_chunk_steps must be a positive integer");
  }
}

} // namespace omni::solver::detail

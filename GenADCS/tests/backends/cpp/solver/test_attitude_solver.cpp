#include "solver/attitude_solver.hpp"

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

int main() {
  const omni::solver::AttitudeSolver solver(0.25);
  if (std::abs(solver.dt() - 0.25) > 1e-15) {
    std::cerr << "attitude solver dt mismatch\n";
    return 1;
  }
  for (const double invalid : {0.0, -1.0,
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
    try {
      (void)omni::solver::AttitudeSolver(invalid);
      std::cerr << "invalid dt accepted\n";
      return 1;
    } catch (const std::invalid_argument &) {
    }
  }
  return 0;
}

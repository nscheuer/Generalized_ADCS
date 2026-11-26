#include "solver/orbital_solver.hpp"
#include <stdexcept>
#include <utility>
struct Propagator {
  double step_size;
  double dt() const { return step_size; }
  int step(int state, int environment) const { return state + environment; }
  int integrate(int state, int tf) const { return state * tf; }
};
int main() {
  omni::solver::OrbitalSolver<Propagator> solver(Propagator{12.5});
  if (solver.dt() != 12.5 || solver.propagator.step_size != 12.5)
    throw std::runtime_error("orbital solver storage or dt failed");
  if (solver.step(3, 4) != 7 || solver.integrate(3, 4) != 12)
    throw std::runtime_error("orbital solver forwarding failed");
}

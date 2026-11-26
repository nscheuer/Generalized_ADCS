#pragma once
#include <utility>

namespace omni::solver {
// Native solver that owns an orbital propagator and forwards its propagation API.
template <typename Propagator> struct OrbitalSolver {
  Propagator propagator;
  explicit OrbitalSolver(Propagator value) : propagator(std::move(value)) {}
  auto dt() const -> decltype(propagator.dt()) { return propagator.dt(); }
  template <typename... Args> decltype(auto) step(Args &&...args) const {
    return propagator.step(std::forward<Args>(args)...);
  }
  template <typename... Args> decltype(auto) integrate(Args &&...args) const {
    return propagator.integrate(std::forward<Args>(args)...);
  }
};
} // namespace omni::solver

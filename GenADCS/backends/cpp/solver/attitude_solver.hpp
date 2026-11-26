#pragma once

#include "../propagation/coupled_propagators/_coupled_dynamics.hpp"
#include "../propagation/coupled_propagators/integrate.hpp"
#include "../propagation/attitude_propagators/reference_attitude_propagator.hpp"
#include <Eigen/Dense>
#include <string>
#include <variant>

namespace omni::solver {

struct RK4AttitudeConfiguration {
  double dt;
};

// Owns a native attitude propagator and forwards solver operations to it.
class AttitudeSolver {
public:
  explicit AttitudeSolver(double step_seconds);
  AttitudeSolver(double step_seconds, const std::string &mode);
  double dt() const noexcept;
  bool is_reference() const noexcept;
  int mode() const noexcept;

  propagation::Trajectory integrate_rk4(
      const propagation::Vector &state, double initial_julian_date, double tf,
      const propagation::Runtime &runtime, std::size_t record_every) const;
  void apply_reference(Eigen::Ref<Eigen::MatrixXd> states,
                       const std::vector<double> &times,
                       double initial_julian_date, double mu) const;

private:
  std::variant<RK4AttitudeConfiguration,
               propagation::ReferenceAttitudePropagator>
      propagator_;
};

} // namespace omni::solver

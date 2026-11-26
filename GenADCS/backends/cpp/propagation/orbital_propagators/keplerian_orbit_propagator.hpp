#pragma once
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <functional>

namespace omni::propagation {
Eigen::Vector3d keplerian_position_velocity_step(const Eigen::Vector3d &position,
                                                  const Eigen::Vector3d &velocity,
                                                  double mu, double dt,
                                                  Eigen::Vector3d &next_velocity);

class KeplerianOrbitPropagator {
public:
  explicit KeplerianOrbitPropagator(double dt);
  double dt() const noexcept { return dt_; }
  Eigen::VectorXd step(const Eigen::VectorXd &state, double mu) const;
  std::pair<std::vector<double>, Eigen::MatrixXd>
  integrate(const Eigen::VectorXd &state, double mu, double tf,
            int record_every = 1,
            const std::function<void(std::size_t)> &progress = {}) const;
private:
  double dt_;
};
} // namespace omni::propagation

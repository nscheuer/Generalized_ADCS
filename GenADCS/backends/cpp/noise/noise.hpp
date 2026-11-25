#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
namespace omni::noise {
class Noise {
public:
  explicit Noise(std::optional<Eigen::MatrixXd> covariance = std::nullopt);
  Eigen::VectorXd apply(const Eigen::VectorXd &value, double dt);
  Eigen::VectorXd sample(std::uint64_t seed, std::uint64_t tick) const;
  std::optional<Eigen::MatrixXd> covariance;
  std::optional<Eigen::MatrixXd> cholesky_factor;
  std::uint64_t seed = 0;
  std::uint64_t tick = 0;
};
} // namespace omni::noise

#include "noise.hpp"
#include "../helpers/random/philox.hpp"
#include <stdexcept>
namespace omni::noise {
Noise::Noise(std::optional<Eigen::MatrixXd> c) : covariance(std::move(c)) {
  if (!covariance)
    return;
  const auto &m = *covariance;
  if (m.rows() != m.cols() || !m.allFinite() || !m.isApprox(m.transpose()))
    throw std::invalid_argument(
        "covariance must be a finite symmetric square matrix");
  Eigen::LLT<Eigen::MatrixXd> factor(m);
  if (factor.info() != Eigen::Success)
    throw std::invalid_argument("covariance must be positive definite");
  cholesky_factor = Eigen::MatrixXd(factor.matrixL());
}
Eigen::VectorXd Noise::sample(std::uint64_t seed_value,
                              std::uint64_t tick_value) const {
  if (!cholesky_factor)
    throw std::logic_error(
        "configure a covariance or use a concrete noise model");
  auto values = omni::helpers::random::standard_normals(
      seed_value, tick_value, 0,
      static_cast<std::uint32_t>(cholesky_factor->rows()));
  return *cholesky_factor *
         Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
}
Eigen::VectorXd Noise::apply(const Eigen::VectorXd &value, double) {
  if (!cholesky_factor)
    throw std::logic_error(
        "configure a covariance or use a concrete noise model");
  if (value.size() != cholesky_factor->rows())
    throw std::invalid_argument("value length must match noise covariance");
  return value + sample(seed, tick++);
}
} // namespace omni::noise

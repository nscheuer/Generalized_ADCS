#include "derivatives.hpp"
#include "dynamics.hpp"
#include "rotation_types.hpp"
#include "vector.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <string>
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) +                      \
                               " failed: " #condition);                        \
  } while (false)
#define CHECK_NEAR(actual, expected, tolerance)                                \
  CHECK(std::abs((actual) - (expected)) <= (tolerance))
namespace math = omni::math;
void test_vector() {
  const Eigen::Vector3d a(1, 2, 3);
  const Eigen::Vector3d b(-4, 5, 2);
  const Eigen::Vector3d product = math::cross(a, b);
  CHECK(product.isApprox(a.cross(b), 1e-14));
  CHECK(std::abs(product.dot(a)) < 1e-14);
  CHECK(std::abs(product.dot(b)) < 1e-14);
  CHECK(math::cross(a, b).isApprox(-math::cross(b, a), 1e-14));
  CHECK(math::cross(a, 3.5 * a).norm() < 1e-14);

  Eigen::VectorXd vector(3);
  vector << 3, 4, 0;
  CHECK_NEAR(math::norm(vector), 5.0, 1e-14);
  CHECK(math::normalize(vector).isApprox(vector / 5.0, 1e-14));
  CHECK(math::unit(vector).isApprox(vector / 5.0, 1e-14));
  CHECK(math::normalize(Eigen::VectorXd::Zero(3)).isZero(0));
  bool zero_unit_rejected = false;
  try {
    (void)math::unit(Eigen::VectorXd::Zero(3));
  } catch (const std::invalid_argument &) {
    zero_unit_rejected = true;
  }
  CHECK(zero_unit_rejected);

  const Eigen::Vector3d skew_input(1.5, -2.0, 4.0);
  const Eigen::Matrix3d skew_matrix = math::skew(skew_input);
  CHECK(skew_matrix.isApprox(-skew_matrix.transpose(), 1e-14));
  CHECK((skew_matrix * b).isApprox(skew_input.cross(b), 1e-14));
  CHECK((skew_matrix * skew_input).norm() < 1e-14);
}
int main() { void test_vector(); }

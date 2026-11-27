#include "check.hpp"
#include "environment/gravity/j_gravity_model.hpp"
#include <cmath>

using namespace omni::environment;

int main() {
  const Eigen::Vector3d position(7'000'000.0, 0.0, 0.0);
  const J2 model;
  const Eigen::Vector3d acceleration = model.evaluate(position);
  CHECK(acceleration.allFinite() && acceleration.norm() > 7.5 &&
        acceleration.norm() < 8.5);
  CHECK(acceleration.x() < 0.0 && std::abs(acceleration.y()) < 1e-14);
  const JGravityModel point_mass(std::vector<int>{});
  const Eigen::Vector3d oblique(7'000'000.0, 0.0, 1'000'000.0);
  CHECK((model.evaluate(oblique) - point_mass.evaluate(oblique)).norm() > 1e-4);
  const Eigen::Vector3d rotated(0.0, 7'000'000.0, 1'000'000.0);
  CHECK(std::abs(model.evaluate(rotated).x()) < 1e-14);

  // Regression test: the latitude term once had the wrong sign, which only
  // shows away from the equator. Compare with the textbook J2 acceleration.
  const double mu = 3.986004418e14, re = 6378137.0, j2 = 1.08262668e-3;
  Eigen::VectorXd j2_only = Eigen::VectorXd::Zero(5);
  j2_only[0] = j2;
  for (const Eigen::Vector3d &p :
       {Eigen::Vector3d(7'000'000.0, 0.0, 1'000'000.0),
        Eigen::Vector3d(4'000'000.0, 3'000'000.0, 4'500'000.0),
        Eigen::Vector3d(-2'500'000.0, 1'500'000.0, -6'000'000.0)}) {
    const double r2 = p.squaredNorm(), r = std::sqrt(r2);
    const double k = 1.5 * j2 * mu * re * re / std::pow(r, 5);
    const double zz = 5.0 * p.z() * p.z() / r2;
    const Eigen::Vector3d expected =
        -mu * p / (r2 * r) + k * Eigen::Vector3d(p.x() * (zz - 1.0),
                                                 p.y() * (zz - 1.0),
                                                 p.z() * (zz - 3.0));
    const Eigen::Vector3d actual =
        gravity_acceleration_inertial_m_s2(p, 0.0, mu, re, j2_only);
    CHECK((actual - expected).cwiseAbs().maxCoeff() < 1e-12);
  }
  return 0;
}

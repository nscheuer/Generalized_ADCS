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
  return 0;
}

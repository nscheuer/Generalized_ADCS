#include "check.hpp"
#include "environment/atmosphere/NRMLSISE00/nrlmsise00.hpp"
#include "environment/atmosphere/atmospheric_model.hpp"
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <stdexcept>

using namespace omni::environment;

int main() {
  const Eigen::Vector3d position(6'778'137.0, 0.0, 0.0);
  const double epoch = 2458849.5;
  const std::array<double, 7> ap_history{-1, -1, -1, -1, -1, -1, -1};
  const double rho =
      nrlmsise_density(position, epoch, 150.0, 150.0, 4.0, true, ap_history);
  CHECK(std::isfinite(rho) && rho > 1e-13 && rho < 1e-9);
  const double higher =
      nrlmsise_density(Eigen::Vector3d(7'000'000.0, 0.0, 0.0), epoch, 150.0,
                       150.0, 4.0, true, ap_history);
  CHECK(rho > higher && higher > 0.0);
  CHECK(nrlmsise_density(position, epoch, 150.0, 150.0, 4.0, true,
                         ap_history) == rho);

  const Eigen::Vector3d velocity =
      co_rotating_atmosphere_velocity(position, epoch);
  CHECK(std::abs(velocity.x()) < 1e-12);
  CHECK(std::abs(velocity.y() - 7.2921150e-5 * position.x()) < 1e-10);
  CHECK(std::abs(velocity.z()) < 1e-12);
  const Eigen::Vector3d omega(4.0, 5.0, 6.0), p(1.0, 2.0, 3.0);
  CHECK((co_rotating_atmosphere_velocity(p, epoch, omega) - omega.cross(p))
            .norm() < 1e-14);

  bool unimplemented = false;
  try {
    (void)atmospheric_density(position, epoch);
  } catch (const std::logic_error &) {
    unimplemented = true;
  }
  CHECK(unimplemented);
  return 0;
}

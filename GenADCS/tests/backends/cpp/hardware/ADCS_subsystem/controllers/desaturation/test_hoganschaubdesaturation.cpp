#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/desaturation/hoganschaubdesaturation.hpp"

#include <Eigen/Core>
#include <limits>
#include <stdexcept>

using namespace omni::hardware::controllers;

int main() {
  bool threw_zero = false;
  try {
    HoganSchaubManagement bad(0.0);
  } catch (const std::invalid_argument &) {
    threw_zero = true;
  }
  CHECK(threw_zero);

  bool threw_nan = false;
  try {
    HoganSchaubManagement bad(0.5, std::numeric_limits<double>::quiet_NaN());
  } catch (const std::invalid_argument &) {
    threw_nan = true;
  }
  CHECK(threw_nan);

  HoganSchaubManagement hs(0.5, 0.1, 0.2);
  CHECK(hs.to_params().isApprox(Eigen::Vector3d(0.5, 0.1, 0.2), 1e-12));
  CHECK(NoDesaturation().to_params().isApprox(Eigen::Vector3d::Zero(), 1e-12));

  const Eigen::VectorXd u0 =
      hogan_schaub_u0(Eigen::Vector3d(0.3, -0.1, 0.1), 0.5, 0.1);
  CHECK(u0.isApprox(Eigen::Vector3d(0.1, -0.1, 0.0), 1e-12));
  return 0;
}

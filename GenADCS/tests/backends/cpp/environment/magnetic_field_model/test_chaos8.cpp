#include "check.hpp"
#include "environment/native_models.hpp"
#include <stdexcept>

using namespace omni::environment;

int main() {
  const Eigen::Vector3d p(6'778'137.0, 0.0, 0.0);
  const CHAOS8 core(false), full(true);
  const Eigen::Vector3d core_field = core.field(p, 2458849.5);
  const Eigen::Vector3d full_field = full.field(p, 2458849.5);
  CHECK(core_field.allFinite() && core_field.norm() > 1e-5 &&
        core_field.norm() < 7e-5);
  CHECK(full_field.allFinite() && (full_field - core_field).norm() > 0.0);
  bool out_of_range = false;
  try {
    (void)core.field(p, 2378496.5);
  } catch (const std::out_of_range &) {
    out_of_range = true;
  }
  CHECK(out_of_range);
  return 0;
}

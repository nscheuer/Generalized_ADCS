#include "check.hpp"
#include "environment/native_models.hpp"
#include <stdexcept>

using namespace omni::environment;

int main() {
  bool unimplemented = false;
  try {
    (void)WMM().field(Eigen::Vector3d(6'778'137.0, 0.0, 0.0), 2458849.5);
  } catch (const std::runtime_error &) {
    unimplemented = true;
  }
  CHECK(unimplemented);
  return 0;
}

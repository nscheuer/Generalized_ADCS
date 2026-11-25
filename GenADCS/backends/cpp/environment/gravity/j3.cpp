#include "j3.hpp"
namespace omni::environment {
J3::J3() : JGravityModel(3) {}
int J3::degree() const noexcept { return 3; }
} // namespace omni::environment

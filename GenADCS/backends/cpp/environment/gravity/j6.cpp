#include "j6.hpp"
namespace omni::environment {
J6::J6() : JGravityModel(6) {}
int J6::degree() const noexcept { return 6; }
} // namespace omni::environment

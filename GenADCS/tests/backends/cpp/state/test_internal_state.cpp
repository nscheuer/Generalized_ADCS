#include "check.hpp"
#include "internal_state.hpp"
#include <limits>

using namespace omni;

int main()
{
    InternalState empty;
    CHECK(empty.wheel_momentum.size() == 0);
    Eigen::VectorXd source(2);
    source << 1.5, -2;
    InternalState state(source);
    CHECK(state.wheel_momentum.size() == 2);
    CHECK(state.wheel_momentum[1] == -2);
    source[0] = 9;
    CHECK(state.wheel_momentum[0] == 1.5); // Eigen storage owns its values.
    WheelMomentumState wheels(state.wheel_momentum);
    CHECK(wheels.wheel_momentum.size() == 2);
    Eigen::VectorXd bad(1);
    bad[0] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] { InternalState invalid(bad); });
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] { InternalState invalid(bad); });
}

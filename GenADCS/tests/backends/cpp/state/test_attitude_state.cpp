#include "attitude_state.hpp"
#include "check.hpp"
#include <cmath>
#include <limits>

using namespace omni;

int main() {
  Vector3 rates;
  rates << 1, 2, 3;
  Quaternion q;
  q << 2, 0, 0, 0;
  AttitudeState state(q, rates);
  CHECK(state.rates == rates);
  CHECK(state.orientation == q);
  CHECK(state.quaternion_array()[0] == 1);
  CHECK(std::abs(state.quaternion_array().norm() - 1) < 1e-12);
  q[0] = 9;
  CHECK(state.orientation[0] == 2); // Native value ownership.
  Quaternion negative;
  negative << -3, 0, 0, 0;
  CHECK(AttitudeState(negative, rates).quaternion_array()[0] == -1);
  expect_throw<std::invalid_argument>(
      [&] { AttitudeState bad(Quaternion::Zero(), rates); });
  Quaternion nan = Quaternion::Zero();
  nan[0] = std::numeric_limits<double>::quiet_NaN();
  expect_throw<std::invalid_argument>([&] { AttitudeState bad(nan, rates); });
  Vector3 inf = rates;
  inf[0] = std::numeric_limits<double>::infinity();
  expect_throw<std::invalid_argument>(
      [&] { AttitudeState bad(negative, inf); });
}

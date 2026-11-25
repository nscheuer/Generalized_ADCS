#include "check.hpp"
#include "dynamics/third_body_model/lunisolar.hpp"
using namespace omni::dynamics;
int main() {
  V p(7e6, 0, 0), sun(1.5e11, 0, 0), moon(0, 3.8e8, 0);
  auto a = lunisolar(p, sun, moon, 1.3271244e20, 4.9048695e12);
  CHECK(a.allFinite() && a.norm() > 1e-8 && a.norm() < 1e-5);
  bool rejected = false;
  try {
    lunisolar(V::Zero(), V::Zero(), moon, 1, 1);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    lunisolar(V::Zero(), sun, V::Zero(), 1, 1);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}

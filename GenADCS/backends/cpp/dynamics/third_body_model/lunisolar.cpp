#include "lunisolar.hpp"
#include <cmath>
#include <stdexcept>
namespace omni::dynamics {
V lunisolar(const V &p, const V &sun, const V &moon, double sun_mu,
            double moon_mu) {
  V result = V::Zero();
  for (const auto &pair :
       {std::pair<V, double>{sun, sun_mu}, {moon, moon_mu}}) {
    V rel = pair.first - p;
    const double d = pair.first.norm(), r = rel.norm();
    if (d == 0 || r == 0)
      throw std::invalid_argument("third-body distance must be nonzero");
    result +=
        pair.second * (rel / std::pow(r, 3) - pair.first / std::pow(d, 3));
  }
  return result;
}
} // namespace omni::dynamics

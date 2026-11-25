#pragma once
#include "../_geometry.hpp"
namespace omni::dynamics {
V lunisolar(const V &p, const V &sun, const V &moon, double sun_mu,
            double moon_mu);
}

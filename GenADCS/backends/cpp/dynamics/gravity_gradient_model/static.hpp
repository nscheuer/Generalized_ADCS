#pragma once
#include "../_geometry.hpp"
namespace omni::dynamics {
V gravity_gradient(const V &p, const Q &q, const M &inertia, double mu);
}

#pragma once
#include "_coupled_dynamics.hpp"
namespace omni::propagation
{
Vector rk4_step(const Vector &, double, double, const Runtime &);
}

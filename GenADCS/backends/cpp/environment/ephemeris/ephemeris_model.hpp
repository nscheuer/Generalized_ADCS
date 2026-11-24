#pragma once
#include "../native_models.hpp"
namespace omni::environment
{
std::tuple<double, bool, bool> eclipse_state(const Eigen::Vector3d &, const Eigen::Vector3d &, double, double);
}

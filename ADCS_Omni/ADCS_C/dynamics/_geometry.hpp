#pragma once
#include <Eigen/Dense>
namespace omni::dynamics {
using V = Eigen::Vector3d;
using Q = Eigen::Vector4d;
using M = Eigen::Matrix3d;
M rotation(const Q& q);
}

#pragma once
#include "../_geometry.hpp"
namespace omni::dynamics
{
V drag(const V &velocity, const Q &q, double density, const Eigen::MatrixXd &normals, const Eigen::VectorXd &areas,
       const Eigen::VectorXd &cd, const Eigen::MatrixXd &arms, bool torque);
}

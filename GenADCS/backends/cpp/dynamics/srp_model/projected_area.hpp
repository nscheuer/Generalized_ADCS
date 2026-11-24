#pragma once
#include "../_geometry.hpp"
namespace omni::dynamics
{
V srp(const V &position, const V &sun, const Q &q, const Eigen::MatrixXd &normals, const Eigen::VectorXd &areas,
      const Eigen::MatrixXd &optical, const Eigen::MatrixXd &arms, double illumination, bool torque);
}

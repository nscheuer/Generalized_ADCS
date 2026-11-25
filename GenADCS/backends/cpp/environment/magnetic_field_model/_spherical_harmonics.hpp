#pragma once
#include <Eigen/Dense>
namespace omni::environment {
Eigen::Vector3d eval_field(const Eigen::Vector3d &, double,
                           const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                           int);
}

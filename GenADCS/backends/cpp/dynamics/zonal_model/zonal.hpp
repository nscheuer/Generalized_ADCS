#pragma once
#include "../_geometry.hpp"
namespace omni::dynamics {
V zonal(const V& position,double mass,double epoch,double mu,double radius,const Eigen::VectorXd& coefficients);
}

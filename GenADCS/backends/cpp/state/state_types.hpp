#ifndef ADCS_OMNI_STATE_TYPES_HPP
#define ADCS_OMNI_STATE_TYPES_HPP

#include <Eigen/Core>

namespace omni {
using Vector3 = Eigen::Vector3d;
using Quaternion = Eigen::Matrix<double, 4, 1>;
using KeplerianElements = Eigen::Matrix<double, 6, 1>;
} // namespace omni

#endif // ADCS_OMNI_STATE_TYPES_HPP

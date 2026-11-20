#include "wmm.hpp"
#include "../../native_helpers.hpp"
#include <stdexcept>
namespace omni::environment {
Eigen::Vector3d WMM::field_inertial_tesla(const Eigen::Vector3d&,double) const { throw std::runtime_error("WMM coefficients are not implemented"); }


}

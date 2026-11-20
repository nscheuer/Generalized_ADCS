#include "igrf14.hpp"
#include "../../native_helpers.hpp"
#include "../../../helpers/math/time/conversions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace omni::environment {
IGRF14::IGRF14() { auto d=read_shc(environment_data_path("magnetic_field_model/igrf14/kernels/IGRF14.shc"),1,13);for(double year:d.times)times_.push_back(omni::time::calendar_to_jd(static_cast<int>(year),1,1));g_=std::move(d.g);h_=std::move(d.h); }
Eigen::Vector3d IGRF14::field_inertial_tesla(const Eigen::Vector3d& p,double jd) const { if(jd<times_.front()||jd>times_.back())throw std::out_of_range("IGRF-14 coefficients cover 1900-01-01 to 2030-01-01");auto upper=std::upper_bound(times_.begin(),times_.end(),jd);std::size_t i=upper==times_.end()?times_.size()-2U:static_cast<std::size_t>(upper-times_.begin()-1);const double w=(jd-times_[i])/(times_[i+1U]-times_[i]);return eval_field(p,jd,g_[i]+w*(g_[i+1U]-g_[i]),h_[i]+w*(h_[i+1U]-h_[i]),13); }

}

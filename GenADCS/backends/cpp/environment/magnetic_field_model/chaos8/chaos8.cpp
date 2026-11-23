#include "chaos8.hpp"
#include "../../native_helpers.hpp"
#include "../../../helpers/math/time/conversions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace omni::environment {
CHAOS8::CHAOS8(bool include_static):include_static_(include_static) { auto core=read_shc(environment_data_path("magnetic_field_model/chaos8/CHAOS-8.7_core.shc"),1,20);times_=std::move(core.times);g_=std::move(core.g);h_=std::move(core.h);static_g_=Eigen::MatrixXd::Zero(186,186);static_h_=static_g_;if(include_static_){auto stat=read_shc(environment_data_path("magnetic_field_model/chaos8/CHAOS-8.7_static.shc"),21,185);static_g_=stat.g[0];static_h_=stat.h[0];} }
Eigen::Vector3d CHAOS8::field_inertial_tesla(const Eigen::Vector3d& p,double jd) const { const double year=2000.0+(jd-2451544.5)/365.25;if(year<times_.front()||year>times_.back())throw std::out_of_range("Epoch outside CHAOS-8.7 core coefficient range");const auto up=std::upper_bound(times_.begin(),times_.end(),year);std::size_t start=(up==times_.begin()?0U:static_cast<std::size_t>(up-times_.begin()-1));start=(start/5U)*5U;start=std::min(start,times_.size()-6U);Eigen::MatrixXd g=Eigen::MatrixXd::Zero(186,186),h=Eigen::MatrixXd::Zero(186,186);if(include_static_){g=static_g_;h=static_h_;}for(std::size_t j=0;j<6U;++j){double w=1.0;for(std::size_t k=0;k<6U;++k)if(j!=k)w*= (year-times_[start+k])/(times_[start+j]-times_[start+k]);g.block(1,0,20,21)+=w*g_[start+j].block(1,0,20,21);h.block(1,0,20,21)+=w*h_[start+j].block(1,0,20,21);}return eval_field(p,jd,g,h,include_static_?185:20); }

}

#include "ephemeris_model.hpp"
#include "../native_helpers.hpp"
#include <algorithm>
#include <cmath>
namespace omni::environment {
constexpr double kPi=3.14159265358979323846;
std::tuple<double,bool,bool> eclipse_state(const Eigen::Vector3d& r,const Eigen::Vector3d& sun,double body_radius,double sun_radius) { const Eigen::Vector3d s=sun-r,b=-r;const double ds=s.norm(),db=b.norm();if(ds<=sun_radius||db<=body_radius)return {0.0,true,false};const double as=std::asin(std::min(1.0,sun_radius/ds)),ab=std::asin(std::min(1.0,body_radius/db));const double sep=std::acos(std::clamp(s.dot(b)/(ds*db),-1.0,1.0));if(sep>=as+ab)return {1.0,false,false};if(ab>=as&&sep<=ab-as)return {0.0,true,false};double area=0.0;if(sep<=std::abs(ab-as))area=kPi*std::min(as,ab)*std::min(as,ab);else {const double x=(sep*sep+as*as-ab*ab)/(2.0*sep*as),y=(sep*sep+ab*ab-as*as)/(2.0*sep*ab);const double rad=std::max(0.0,(-sep+as+ab)*(sep+as-ab)*(sep-as+ab)*(sep+as+ab));area=as*as*std::acos(std::clamp(x,-1.0,1.0))+ab*ab*std::acos(std::clamp(y,-1.0,1.0))-0.5*std::sqrt(rad);}return {std::clamp(1.0-area/(kPi*as*as),0.0,1.0),false,true}; }

}

#include "_coupled_dynamics.hpp"
#include "../../dynamics/drag_model/projected_area.hpp"
#include "../../dynamics/srp_model/projected_area.hpp"
#include "../../dynamics/gravity_gradient_model/static.hpp"
#include "../../dynamics/residual_dipole_model/static.hpp"
#include "../../dynamics/third_body_model/lunisolar.hpp"
#include "../../environment/gravity/j_gravity_model.hpp"
#include "../../environment/atmosphere/atmospheric_model.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include <pybind11/eigen.h>
#include <array>
#include <algorithm>
#include <cmath>
namespace py=pybind11;
namespace omni::propagation {
using V=Eigen::Vector3d;using Q=Eigen::Vector4d;using M=Eigen::Matrix3d;
static thread_local Runtime* active_runtime=nullptr;
Runtime::Runtime(const py::object& p):
 mu(p.attr("mu").cast<double>()),radius(p.attr("radius").cast<double>()),mass(p.attr("mass").cast<double>()),
 inertia(p.attr("inertia").cast<M>()),inverse_inertia(p.attr("inverse_inertia").cast<M>()),inverse_bus_inertia(p.attr("inverse_bus_inertia").cast<M>()),
 zonals(p.attr("zonals").cast<Vector>()),areas(p.attr("areas").cast<Vector>()),drag_coefficients(p.attr("drag_coefficients").cast<Vector>()),
 wheel_inertias(p.attr("wheel_inertias").cast<Vector>()),wheel_commands(p.attr("wheel_commands").cast<Vector>()),wheel_max_momentum(p.attr("wheel_max_momentum").cast<Vector>()),
 normals(p.attr("normals").cast<Eigen::MatrixXd>()),arms(p.attr("arms").cast<Eigen::MatrixXd>()),optical_coefficients(p.attr("optical_coefficients").cast<Eigen::MatrixXd>()),wheel_axes(p.attr("wheel_axes").cast<Eigen::MatrixXd>()),dipole(p.attr("dipole").cast<V>()),
 drag_force(p.attr("drag_force").cast<bool>()),drag_torque(p.attr("drag_torque").cast<bool>()),srp_force(p.attr("srp_force").cast<bool>()),srp_torque(p.attr("srp_torque").cast<bool>()),gravity_gradient(p.attr("gravity_gradient").cast<bool>()),residual_dipole(p.attr("residual_dipole").cast<bool>()),third_body(p.attr("third_body").cast<bool>()),magnetic_include_static(p.attr("magnetic_include_static").cast<bool>()),magnetic_kind(p.attr("magnetic_kind").cast<int>()) {
 if(srp_force||srp_torque||third_body)ephemeris=std::make_unique<omni::environment::DE421>();
 if(residual_dipole){if(magnetic_kind==1)igrf=std::make_unique<omni::environment::IGRF14>();else if(magnetic_kind==2)chaos=std::make_unique<omni::environment::CHAOS8>(magnetic_include_static);else throw py::value_error("unsupported magnetic model");}
}
RuntimeScope::RuntimeScope(const py::object& p):runtime(p),previous(active_runtime){active_runtime=&runtime;}
RuntimeScope::~RuntimeScope(){active_runtime=previous;}
void project(Vector& y,const py::object& p){std::unique_ptr<Runtime> fallback;if(!active_runtime)fallback=std::make_unique<Runtime>(p);const Runtime& r=active_runtime?*active_runtime:*fallback;for(Eigen::Index i=0;i<r.wheel_max_momentum.size();++i)y[13+i]=std::clamp(y[13+i],-r.wheel_max_momentum[i],r.wheel_max_momentum[i]);}
Vector derivative(const Vector& y,double jd,const py::object& p){
 std::unique_ptr<Runtime> fallback;if(!active_runtime)fallback=std::make_unique<Runtime>(p);const Runtime& r=active_runtime?*active_runtime:*fallback;
 Q q=y.segment<4>(0);q.normalize();const V omega=y.segment<3>(4),pos=y.segment<3>(7),vel=y.segment<3>(10);
 double density=0,illumination=0;V relative=vel,sun=V::Zero(),moon=V::Zero(),field=V::Zero();
 if(r.drag_force||r.drag_torque){std::array<double,7> history;history.fill(-1.);density=omni::environment::nrlmsise_density(pos,jd,150.,150.,4.,true,history);relative-=omni::environment::co_rotating_atmosphere_velocity(pos,jd);}
 if(r.srp_force||r.srp_torque||r.third_body){const double tdb=omni::time::utc_jd_to_tdb_jd(jd);sun=1000.*r.ephemeris->position(0,tdb);if(r.third_body)moon=1000.*r.ephemeris->position(2,tdb);if(r.srp_force||r.srp_torque)illumination=std::get<0>(omni::environment::eclipse_state(pos,sun,r.radius,696340000.));}
 if(r.residual_dipole)field=r.igrf?r.igrf->field_inertial_tesla(pos,jd):r.chaos->field_inertial_tesla(pos,jd);
 V accel=omni::environment::gravity_acceleration_inertial_m_s2(pos,jd,r.mu,r.radius,r.zonals),force=V::Zero(),torque=V::Zero();
 if(r.drag_force)force+=omni::dynamics::drag(relative,q,density,r.normals,r.areas,r.drag_coefficients,r.arms,false);
 if(r.drag_torque)torque+=omni::dynamics::drag(relative,q,density,r.normals,r.areas,r.drag_coefficients,r.arms,true);
 if(r.srp_force)force+=omni::dynamics::srp(pos,sun,q,r.normals,r.areas,r.optical_coefficients,r.arms,illumination,false);
 if(r.srp_torque)torque+=omni::dynamics::srp(pos,sun,q,r.normals,r.areas,r.optical_coefficients,r.arms,illumination,true);
 if(r.third_body)accel+=omni::dynamics::lunisolar(pos,sun,moon,1.32712440018e20,4.9048695e12);
 if(r.gravity_gradient)torque+=omni::dynamics::gravity_gradient(pos,q,r.inertia,r.mu);
 if(r.residual_dipole)torque+=omni::dynamics::residual_dipole(r.dipole,field,q);
 for(Eigen::Index i=0;i<r.wheel_commands.size();++i)torque+=r.wheel_axes.row(i).transpose()*r.wheel_commands[i];
 V h=r.inertia*omega;for(Eigen::Index i=0;i<r.wheel_commands.size();++i)h+=r.wheel_axes.row(i).transpose()*y[13+i];
 V omegadot=(r.wheel_commands.size()?r.inverse_bus_inertia:r.inverse_inertia)*(torque-omega.cross(h));
 Vector dy=Vector::Zero(y.size());dy[0]=0.5*q.tail<3>().dot(omega);dy[1]=-0.5*(q[0]*omega[0]+omega[1]*q[3]-omega[2]*q[2]);dy[2]=-0.5*(q[0]*omega[1]+omega[2]*q[1]-omega[0]*q[3]);dy[3]=-0.5*(q[0]*omega[2]+omega[0]*q[2]-omega[1]*q[1]);dy.segment<3>(4)=omegadot;dy.segment<3>(7)=vel;dy.segment<3>(10)=accel+force/r.mass;
 for(Eigen::Index i=0;i<r.wheel_commands.size();++i){double rate=-r.wheel_commands[i]-r.wheel_inertias[i]*r.wheel_axes.row(i).dot(omegadot);if((y[13+i]>=r.wheel_max_momentum[i]&&rate>0)||(y[13+i]<=-r.wheel_max_momentum[i]&&rate<0))rate=0;dy[13+i]=rate;}
 return dy;
}
}

#include "_coupled_dynamics.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <cmath>
namespace py=pybind11;
namespace omni::propagation {
using V=Eigen::Vector3d;using Q=Eigen::Vector4d;using M=Eigen::Matrix3d;
static py::object envmod(){return py::module_::import("ADCS_Omni.ADCS_C.environment._environment");}
static py::object dynmod(){return py::module_::import("ADCS_Omni.ADCS_C.dynamics._dynamics");}
void project(Vector& y,const py::object& p){auto limits=p.attr("wheel_max_momentum").cast<Eigen::VectorXd>();for(Eigen::Index i=0;i<limits.size();++i)y[13+i]=std::max(-limits[i],std::min(limits[i],y[13+i]));}
Vector derivative(const Vector& y,double jd,const py::object& p){
 const py::object env=envmod(),dyn=dynmod();Q q=y.segment<4>(0);q.normalize();const V omega=y.segment<3>(4),pos=y.segment<3>(7),vel=y.segment<3>(10);const double mu=p.attr("mu").cast<double>(),radius=p.attr("radius").cast<double>(),mass=p.attr("mass").cast<double>();const M inertia=p.attr("inertia").cast<M>();const Eigen::MatrixXd axes=p.attr("wheel_axes").cast<Eigen::MatrixXd>();const Eigen::VectorXd commands=p.attr("wheel_commands").cast<Eigen::VectorXd>();
 const bool df=p.attr("drag_force").cast<bool>(),dt=p.attr("drag_torque").cast<bool>(),sf=p.attr("srp_force").cast<bool>(),st=p.attr("srp_torque").cast<bool>(),gg=p.attr("gravity_gradient").cast<bool>(),rd=p.attr("residual_dipole").cast<bool>(),tb=p.attr("third_body").cast<bool>();
 double density=0,illumination=0;V relative=vel,sun=V::Zero(),moon=V::Zero(),field=V::Zero();
 if(df||dt){density=env.attr("nrlmsise00_density")(pos,jd).cast<double>();relative-=env.attr("co_rotating_atmosphere_velocity")(pos,jd).cast<V>();}
 if(sf||st||tb){double tdb=py::module_::import("ADCS_Omni.ADCS_C.helpers.math.time.conversions").attr("utc_jd_to_tdb_jd")(jd).cast<double>();py::object eph=env.attr("DE421")();sun=1000.0*eph.attr("position")(0,tdb).cast<V>();if(tb)moon=1000.0*eph.attr("position")(2,tdb).cast<V>();if(sf||st)illumination=env.attr("eclipse_state")(pos,sun,radius,696340000.0).cast<py::tuple>()[0].cast<double>();}
 if(rd){const int kind=p.attr("magnetic_kind").cast<int>();field=(kind==1?env.attr("IGRF14")():env.attr("CHAOS8")(p.attr("magnetic_include_static"))).attr("field_inertial_tesla")(pos,jd).cast<V>();}
 V accel=env.attr("gravity_acceleration_inertial_m_s2")(pos,jd,mu,radius,p.attr("zonals")).cast<V>();V force=V::Zero(),torque=V::Zero();const py::object normals=p.attr("normals"),arms=p.attr("arms"),areas=p.attr("areas"),cd=p.attr("drag_coefficients"),optical=p.attr("optical_coefficients");
 if(df)force+=dyn.attr("projected_area_drag_force")(relative,q,density,normals,areas,cd).cast<V>();if(dt)torque+=dyn.attr("projected_area_drag_torque")(relative,q,density,normals,arms,areas,cd).cast<V>();
 if(sf)force+=dyn.attr("projected_area_srp_force")(pos,sun,q,normals,areas,optical,illumination).cast<V>();if(st)torque+=dyn.attr("projected_area_srp_torque")(pos,sun,q,normals,arms,areas,optical,illumination).cast<V>();
 if(tb)accel+=dyn.attr("lunisolar_acceleration")(pos,sun,moon,1.32712440018e20,4.9048695e12).cast<V>();if(gg)torque+=dyn.attr("static_gravity_gradient_torque")(pos,q,inertia,mu).cast<V>();if(rd)torque+=dyn.attr("static_residual_dipole_torque")(p.attr("dipole"),field,q).cast<V>();for(Eigen::Index i=0;i<commands.size();++i)torque+=axes.row(i).transpose()*commands[i];
 V h=inertia*omega;for(Eigen::Index i=0;i<commands.size();++i)h+=axes.row(i).transpose()*y[13+i];M inverse=(commands.size()?p.attr("inverse_bus_inertia"):p.attr("inverse_inertia")).cast<M>();V omegadot=inverse*(torque-omega.cross(h));Vector dy=Vector::Zero(y.size());dy[0]=0.5*q.tail<3>().dot(omega);dy[1]=-0.5*(q[0]*omega[0]+omega[1]*q[3]-omega[2]*q[2]);dy[2]=-0.5*(q[0]*omega[1]+omega[2]*q[1]-omega[0]*q[3]);dy[3]=-0.5*(q[0]*omega[2]+omega[0]*q[2]-omega[1]*q[1]);dy.segment<3>(4)=omegadot;dy.segment<3>(7)=vel;dy.segment<3>(10)=accel+force/mass;
 const auto inertias=p.attr("wheel_inertias").cast<Eigen::VectorXd>(),limits=p.attr("wheel_max_momentum").cast<Eigen::VectorXd>();for(Eigen::Index i=0;i<commands.size();++i){double rate=-commands[i]-inertias[i]*axes.row(i).dot(omegadot);if((y[13+i]>=limits[i]&&rate>0)||(y[13+i]<=-limits[i]&&rate<0))rate=0;dy[13+i]=rate;}return dy;
}
}

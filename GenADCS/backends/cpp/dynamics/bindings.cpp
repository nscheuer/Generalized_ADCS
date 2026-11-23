#include "drag_model/projected_area.hpp"
#include "srp_model/projected_area.hpp"
#include "gravity_gradient_model/static.hpp"
#include "residual_dipole_model/static.hpp"
#include "third_body_model/lunisolar.hpp"
#include "zonal_model/zonal.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
using omni::dynamics::V; using omni::dynamics::Q; using omni::dynamics::M;
PYBIND11_MODULE(_dynamics,m){
 m.def("projected_area_drag_force",[](const V& v,const Q& q,double rho,const Eigen::MatrixXd& n,const Eigen::VectorXd& a,const Eigen::VectorXd& cd){
     if(n.rows()!=a.size()||cd.size()!=a.size()) throw py::value_error("panel arrays must have matching row counts");
     return omni::dynamics::drag(v,q,rho,n,a,cd,Eigen::MatrixXd(),false);
 });
 m.def("projected_area_drag_torque",[](const V& v,const Q& q,double rho,const Eigen::MatrixXd& n,const Eigen::MatrixXd& arms,const Eigen::VectorXd& a,const Eigen::VectorXd& cd){
     if(n.rows()!=a.size()||arms.rows()!=a.size()||cd.size()!=a.size()) throw py::value_error("panel arrays must have matching row counts");
     return omni::dynamics::drag(v,q,rho,n,a,cd,arms,true);
 });
 m.def("projected_area_srp_force",[](const V& p,const V& s,const Q& q,const Eigen::MatrixXd& n,const Eigen::VectorXd& a,const Eigen::MatrixXd& optical,double light){return omni::dynamics::srp(p,s,q,n,a,optical,Eigen::MatrixXd(),light,false);},py::arg("position_i"),py::arg("sun_position_i"),py::arg("q_b_from_i"),py::arg("normals_b"),py::arg("areas"),py::arg("optical_coefficients"),py::arg("illumination")=1.0);
 m.def("projected_area_srp_torque",[](const V& p,const V& s,const Q& q,const Eigen::MatrixXd& n,const Eigen::MatrixXd& arms,const Eigen::VectorXd& a,const Eigen::MatrixXd& optical,double light){return omni::dynamics::srp(p,s,q,n,a,optical,arms,light,true);},py::arg("position_i"),py::arg("sun_position_i"),py::arg("q_b_from_i"),py::arg("normals_b"),py::arg("arms_b"),py::arg("areas"),py::arg("optical_coefficients"),py::arg("illumination")=1.0);
 m.def("static_residual_dipole_torque",[](const V& dipole,const V& field,const Q& q){return omni::dynamics::residual_dipole(dipole,field,q);});
 m.def("static_gravity_gradient_torque",[](const V& p,const Q& q,const M& inertia,double mu){return omni::dynamics::gravity_gradient(p,q,inertia,mu);});
 m.def("lunisolar_acceleration",[](const V& p,const V& sun,const V& moon,double sun_mu,double moon_mu){return omni::dynamics::lunisolar(p,sun,moon,sun_mu,moon_mu);});
 m.def("zonal_gravity_force",&omni::dynamics::zonal);
}

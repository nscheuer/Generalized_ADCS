#include "keplerian_orbit_propagator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace omni::propagation {
namespace {
std::pair<double,double> stumpff(double z) {
  if (z > 1e-8) { const double x=std::sqrt(z); return {(1-std::cos(x))/z,(x-std::sin(x))/(x*x*x)}; }
  if (z < -1e-8) { const double x=std::sqrt(-z); return {(std::cosh(x)-1)/(-z),(std::sinh(x)-x)/(x*x*x)}; }
  return {0.5-z/24+z*z/720,1.0/6-z/120+z*z/5040};
}
}
Eigen::Vector3d keplerian_position_velocity_step(const Eigen::Vector3d &r,
 const Eigen::Vector3d &v, double mu, double dt, Eigen::Vector3d &vout) {
 if (!std::isfinite(mu) || mu<=0) throw std::invalid_argument("gravitational parameter must be finite and positive");
 const double radius=r.norm(); if (!std::isfinite(radius)||radius<=0) throw std::invalid_argument("orbital position must have positive radius");
 const double root=std::sqrt(mu), radial=r.dot(v)/radius, alpha=2/radius-v.squaredNorm()/mu;
 double chi=root*dt*(std::abs(alpha)>1e-12?std::abs(alpha):1/radius);
 bool converged=false;
 for(int i=0;i<100;++i){ const double z=alpha*chi*chi; auto [c,s]=stumpff(z);
  const double residual=radius*radial/root*chi*chi*c+(1-alpha*radius)*chi*chi*chi*s+radius*chi-root*dt;
  const double derivative=radius*radial/root*chi*(1-z*s)+(1-alpha*radius)*chi*chi*c+radius;
  const double change=residual/derivative; chi-=change;
  if(std::abs(change)<=1e-12*std::max(1.0,std::abs(chi))){converged=true;break;}
 }
 if(!converged) throw std::runtime_error("Keplerian universal anomaly did not converge");
 const double z=alpha*chi*chi; auto [c,s]=stumpff(z); const double f=1-chi*chi*c/radius, g=dt-chi*chi*chi*s/root;
 const Eigen::Vector3d rout=f*r+g*v; const double rn=rout.norm();
 const double fd=root/(rn*radius)*(z*s-1)*chi, gd=1-chi*chi*c/rn; vout=fd*r+gd*v; return rout;
}
KeplerianOrbitPropagator::KeplerianOrbitPropagator(double dt):dt_(dt){if(!std::isfinite(dt)||dt<=0)throw std::invalid_argument("dt must be finite and positive seconds");}
Eigen::VectorXd KeplerianOrbitPropagator::step(const Eigen::VectorXd &y,double mu) const {
 if(y.size()<13) throw std::invalid_argument("packed state must contain at least 13 values");
 Eigen::Vector3d vout; auto r=keplerian_position_velocity_step(y.segment<3>(7),y.segment<3>(10),mu,dt_,vout);
 Eigen::VectorXd out=y; out.segment<3>(7)=r; out.segment<3>(10)=vout; return out;
}
std::pair<std::vector<double>,Eigen::MatrixXd> KeplerianOrbitPropagator::integrate(const Eigen::VectorXd &initial,double mu,double tf,int record,const std::function<void(std::size_t)> &progress) const {
 if(!std::isfinite(tf)||tf<0) throw std::invalid_argument("tf must be finite and nonnegative seconds");
 if(record<1) throw std::invalid_argument("record_every must be a positive integer");
 if(initial.size()<13) throw std::invalid_argument("packed state must contain at least 13 values");
 const int full=static_cast<int>(std::floor(tf/dt_)); double rem=tf-full*dt_; if(rem<=1e-12*std::max(1.0,tf)) rem=0;
 const int total=full+(rem>0); std::vector<int> rows{0}; for(int k=record;k<=total;k+=record) rows.push_back(k); if(rows.back()!=total) rows.push_back(total);
 std::vector<double> times; Eigen::MatrixXd output(rows.size(),initial.size()); Eigen::VectorXd y=initial; output.row(0)=y.transpose(); times.push_back(0);
 for(int k=1;k<=total;++k){ Eigen::Vector3d vout; double h=(k==full+1&&rem>0)?rem:dt_; auto r=keplerian_position_velocity_step(y.segment<3>(7),y.segment<3>(10),mu,h,vout); y.segment<3>(7)=r;y.segment<3>(10)=vout;
  if(std::find(rows.begin()+1,rows.end(),k)!=rows.end()){int row=static_cast<int>(times.size());output.row(row)=y.transpose();times.push_back(k==total?tf:std::min(k*dt_,tf));}
  if(progress) progress(1);
 }
 return {times,output};
}
} // namespace omni::propagation

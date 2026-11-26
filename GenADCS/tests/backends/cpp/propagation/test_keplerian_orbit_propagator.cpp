#include "propagation/orbital_propagators/keplerian_orbit_propagator.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(){
 using omni::propagation::KeplerianOrbitPropagator;
 constexpr double mu=3.986004418e14, radius=7.0e6;
 const double speed=std::sqrt(mu/radius), period=2*std::acos(-1.0)*std::sqrt(radius*radius*radius/mu);
 Eigen::VectorXd y=Eigen::VectorXd::Zero(13); y[0]=1;y[7]=radius;y[11]=speed;
 KeplerianOrbitPropagator p(period); auto z=p.step(y,mu);
 if(std::abs(z[7]-radius)>1e-5||std::abs(z[11]-speed)>1e-8)return 1;
 auto [times,states]=KeplerianOrbitPropagator(10).integrate(y,mu,42.5,3);
 if(times.size()!=3||std::abs(times[1]-30)>1e-12||std::abs(times[2]-42.5)>1e-12)return 2;
 Eigen::Vector3d vout; auto r=omni::propagation::keplerian_position_velocity_step(y.segment<3>(7),y.segment<3>(10),mu,42.5,vout);
 if((states.row(2).segment<3>(7).transpose()-r).norm()>1e-6)return 3;
 try {KeplerianOrbitPropagator bad(0); return 4;} catch(const std::invalid_argument&){}
 std::cout<<"Keplerian propagation passed\n";
}

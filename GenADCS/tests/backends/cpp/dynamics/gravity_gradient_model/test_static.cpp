#include "check.hpp"
#include "dynamics/gravity_gradient_model/static.hpp"
using namespace omni::dynamics;
int main(){M I=M::Zero();I.diagonal()<<2,3,4;Q q(1,0,0,0);V p(7e6,7e6,0);auto t=gravity_gradient(p,q,I,3.986e14);CHECK(t.allFinite()&&t.norm()>0);CHECK(gravity_gradient(p,q,4*M::Identity(),3.986e14).norm()==0);bool rejected=false;try{gravity_gradient(V::Zero(),q,I,3.986e14);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);}

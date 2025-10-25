#ifndef TPR_GENERALUTIL_HPP
#define TPR_GENERALUTIL_HPP

#include <armadillo>
#include <tuple>
#include <assert.h>
#include "../ArmaNumpy.hpp"
#include <string>
//namespace py = pybind11;

#define earth_mu (double) 3.986e14
#define EPSVAR (double) 2.22e-16


arma::mat::fixed<4,3> findWMat( arma::vec4 qk);

arma::mat::fixed<3,4>  dRTBdq(const arma::vec4 q, const arma::vec3 B);
arma::mat::fixed<3,3>  dRTBdqQ(const arma::vec4 q, const arma::vec3 B);

arma::mat::fixed<3,3> ddvTRTudqQ(const arma::vec4 q,const  arma::vec3 v,const  arma::vec3 u);
arma::mat::fixed<4,4> ddvTRTudq(const arma::vec4 q,const  arma::vec3 v,const  arma::vec3 u);
arma::cube::fixed<3,3,3> ddRTudqQ(const arma::vec4 q, const arma::vec3 u);
arma::mat33 rotMat(arma::vec4 q);

double sigmoid(double val);
double swish( double val, double beta = 1e8);
double shifted_swish( double val, double limit = 0.0, double beta = 1e8);
double shifted_softplus(double val, double limit = 0.0, double k = 1e6);
double softplus(double val, double k = 1e6);
double smoothstep(double val);
double smoothstep_deriv(double val);
double smoothstep_deriv2(double val);
double softplus_deriv(double val, double k=1e6);
double softplus_deriv2(double val, double k=1e6);
double shifted_softplus_deriv(double val, double limit = 0.0, double k=1e6);
double shifted_softplus_deriv2(double val, double limit = 0.0, double k=1e6);



double cost2AngOnly(arma::vec4 q, arma::vec3 v, arma::vec3 u);
std::tuple<double, arma::vec4,arma::mat44> cost2ang(arma::vec4 q, arma::vec3 s, arma::vec3 e);
std::tuple<double, arma::vec3,arma::mat33> cost2angQ(arma::vec4 q, arma::vec3 s, arma::vec3 e);



arma::cube cubeTimesMat(arma::cube c, arma::mat m);
arma::cube matTimesCube(arma::mat m, arma::cube c);
arma::cube matTimesCubeT(arma::mat m, arma::cube c);
arma::cube matOverCube(arma::mat m, arma::cube c);
arma::mat vecOverCube(arma::vec v, arma::cube c);
arma::mat vecTimesCube(arma::vec v, arma::cube c);


std::tuple<arma::vec4,arma::vec4> quatSetBasis(arma::vec3 v0, arma::vec3 u0);
arma::vec4 closestQuat(arma::vec4 q, arma::vec4 x, arma::vec4 y);
arma::vec4 closestQuatForVecPoint(arma::vec4 q, arma::vec3 v0, arma::vec4 u0);
arma::vec4 quatinv(arma::vec4 q);
arma::vec4 quatmult(arma::vec4 p,arma::vec4 q);
arma::vec4 quaterr(arma::vec4 p,arma::vec4 q);
arma::vec4 normquaterr(arma::vec4 p,arma::vec4 q);


arma::mat33 skewSymmetric(arma::vec3 vk);

#endif

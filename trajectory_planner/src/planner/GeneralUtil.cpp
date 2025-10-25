#include "GeneralUtil.hpp"

//
// static const int CONSTRAINT_NUM = 14;
// static const int INEQ_NUM = 14;
// static const int CONTROL_NUM = 3;
// static const int  STATE_NUM = 7;
// static const int REDUCED_STATE_NUM = 6;
//
// inline const static int EQ_NUM = CONSTRAINT_NUM-INEQ_NUM;

using namespace std;
using namespace arma;


static const double acos_limitL = 1-2e-10;
static const double acos_limitH = 1-1e-10;


arma::cube cubeTimesMat(cube c, mat m)
{
  cube out = cube(c.n_rows,m.n_cols,c.n_slices).zeros();
  for(int j = 0; j<c.n_slices;j++){
    out.slice(j) = c.slice(j)*m;
  }
  return out;
}

arma::cube matTimesCube(mat m, cube c)
{
  cube out = cube(m.n_rows,c.n_cols,c.n_slices).zeros();
  for(int j = 0; j<c.n_slices;j++){
    out.slice(j) = m*c.slice(j);
  }
  return out;
}


arma::cube matTimesCubeT(mat m, cube c)
{
  cube out = cube(m.n_rows,c.n_rows,c.n_slices).zeros();
  for(int j = 0; j<c.n_slices;j++){
    out.slice(j) = m*c.slice(j).t();
  }
  return out;
}

arma::cube matOverCube(mat m, cube c)
{
  // cout<<c<<"\n";
  cube out = cube(c.n_rows,c.n_cols,m.n_rows).zeros();
  for(int j = 0; j<c.n_cols;j++){
    out.col(j) = (m*mat(c.col(j)).t()).t();
  }
  return out;
}

arma::mat vecOverCube(vec v, cube c)
{
  // cout<<c<<"\n";
  mat out = mat(c.n_rows,c.n_cols).zeros();
  for(int j = 0; j<c.n_slices;j++){
    out += v(j)*c.slice(j);
  }
  return out;
}

/*
given v0 (a body frame vector) and u0 (global frame vector), this returns x and y, two quaternions.
any quaternion of the form x*cos(a)+y*sin(a) will align v with u
*/
std::tuple<vec4,vec4> quatSetBasis(vec3 v0, vec3 u0)
{
  vec3 v = normalise(v0);
  vec3 u = normalise(u0);
  // double dvu_denom = pow(2.0*(1.0+dot(v, u)), 0.5);
  // if (abs(dvu_denom)<1e-10)
  // {
  //   dvu_denom = dvu_denom + EPSVAR;
  // }
  // vec4 x = join_cols(vec({dvu_denom/2.0}), cross(v, u)/dvu_denom);
  // vec4 y = join_cols(vec({0.0}), (u+v)/dvu_denom);
  vec4 x = join_cols(vec({1.0+dot(v, u)}), cross(v, u));
  vec4 y = join_cols(vec({0.0}), (u+v));
  x = normalise(x);
  y = normalise(y);
  return make_tuple(x,y);
}
/*
given an aribitrary quaternion q, as well as the two quaternions x and y which describe the goal set,
this returns the quaternion in the goal set with minimal angular distance from q.
*/
vec4 closestQuat(vec4 q, vec4 x, vec4 y){
  q = normalise(q);
  x = normalise(x);
  y = normalise(y);
  double qdx = dot(q,x);//laVec(0);
  double qdy = dot(q,y);//lbVec(0);
  vec4 res = normalise(qdx*x+qdy*y);
  return res*sign(norm_dot(res,q));
}

vec4 quatinv(vec4 q){
  return join_cols(vec({q(0)}),-q.tail(3));
}

vec4 quatmult(vec4 p,vec4 q){
  return join_cols(vec(1).ones()*(p(0)*q(0)-dot(p.tail(3),q.tail(3))),p(0)*q.tail(3) + q(0)*p.tail(3) + cross(p.tail(3),q.tail(3)));
}

vec4 quaterr(vec4 p,vec4 q){
  vec4 pi = quatinv(p);
  return quatmult(pi,q);
}

vec4 normquaterr(vec4 p, vec4 q){
  return normalise(quaterr(normalise(p),normalise(q)));
}


vec4 closestQuatForVecPoint(vec4 q, vec3 v0, vec4 u0){
  std::tuple<vec4,vec4> xy = quatSetBasis(v0, u0);
  return closestQuat(q,get<0>(xy),get<1>(xy));
}


/* This function returns a matrix with -q(2:4)^T as the first row, and q(1) times
the skew symmetric matrix of q(2:4) as the second through fourth rows.
  Arguments:
    Matrix<double, 4, 1> *qk should be a pointer referencing a 4 x 1 matrix
  Returns:
    Matrix<double, 4, 3> wMat
*/
mat::fixed<4,3> findWMat(vec4 qk)
{
  double q0 = qk(0);
  double q1 = qk(1);
  double q2 = qk(2);
  double q3 = qk(3);
  //initializer_list<initializer_list<double>> skewSymexpected_contents = {{0, -v3, v2}, {v3, 0, -v1}, {-v2, v1, 0}};

  //mat vkSkewSym = skewSymmetric(vk);
  return mat({{-q1, -q2,  -q3},
              {q0,  -q3,  q2},
              {q3,  q0,   -q1},
              {-q2, q1,   q0}});
}



/* This function gets the derivative of the ECI-to-body rotation matrix R^T with respect to q1, q2, q3, and q4
   Inputs:
    q = quaternion
   Outputs:
    dRtdq1, dRtdq2, dRtdq3, dRtdq4 = partial derivatives of R^T wrt q1...4
*/
mat::fixed<3,4> dRTBdq(const vec4 q, const vec3 B)
{

  vec3 qv = q.tail(3);//*-1;
  double q0 = q(0);
  return 2.0*join_rows(q0*B - cross(qv,B),mat33().eye()*dot(qv,B) + qv*B.t() - B*qv.t() + q0*skewSymmetric(B));
}

mat::fixed<3,3> dRTBdqQ(const vec4 q, const vec3 B)
{
  return dRTBdq(q,B)*findWMat(q);
}


/* This function gets the derivative of the ECI-to-body rotation matrix R^T with respect to q1, q2, q3, and q4
   Inputs:
    q = quaternion
   Outputs:
    dRtdq1, dRtdq2, dRtdq3, dRtdq4 = partial derivatives of R^T wrt q1...4
*/


mat33 ddvTRTudqQ(const vec4 q, const vec3 v, const vec3 u)
{
  mat::fixed<4,3> Wq = findWMat(q);
  vec3 vxu = cross(v,u);
  mat44 hess = mat44().zeros();
  hess += dot(v,u)*diagmat(vec({1.0,-1.0,-1.0,-1.0}));
  hess(span(0,0),span(1,3)) = vxu.t();
  hess(span(1,3),span(0,0)) = vxu;
  hess(span(1,3),span(1,3)) += v*u.t() + u*v.t();
  hess *= 2.0;

  return Wq.t()*hess*Wq - mat33().eye()*as_scalar(v.t()*dRTBdq(q,u)*q);
}


mat44 ddvTRTudq(const vec4 q, const vec3 v, const vec3 u)
{
  //mat::fixed<4,3> Wq = findWMat(q);
  vec3 vxu = cross(v,u);
  mat44 hess = mat44().zeros();
  hess += dot(v,u)*diagmat(vec({1.0,-1.0,-1.0,-1.0}));
  hess(span(0,0),span(1,3)) = vxu.t();
  hess(span(1,3),span(0,0)) = vxu;
  hess(span(1,3),span(1,3)) += v*u.t() + u*v.t();
  hess *= 2.0;

  return hess;
}

cube::fixed<3,3,3> ddRTudqQ(const vec4 q,const vec3 u)
{
  vec3 x = vec({1.0,0.0,0.0});
  vec3 y = vec({0.0,1.0,0.0});
  vec3 z = vec({0.0,0.0,1.0});
  return join_slices(ddvTRTudqQ(q,x,u),join_slices(ddvTRTudqQ(q,y,u),ddvTRTudqQ(q,z,u)));
}

/* This function gets the ECI-to-body rotation matrix for a quaternion q
   Inputs:
    q = quaternion
  Outputs:
    R = rotation matrix
*/
mat33 rotMat(vec4 q)
{
  //Get q0...q3
  double q0 = q(0);
  double q1 = q(1);
  double q2 = q(2);
  double q3 = q(3);

  double q00 = q0*q0;
  double q01 = q0*q1;
  double q02 = q0*q2;
  double q03 = q0*q3;

  double q11 = q1*q1;
  double q12 = q1*q2;
  double q13 = q1*q3;

  double q22 = q2*q2;
  double q23 = q2*q3;

  double q33 = q3*q3;

  return mat({{q00+q11-q22-q33, 2.0*(q12-q03), 2.0*(q13+q02)},
              {2.0*(q12+q03), q00-q11+q22-q33, 2.0*(q23-q01)},
              {2.0*(q13-q02), 2.0*(q23+q01), q00-q11-q22+q33}});

  //return mat({{q0*q0+q1*q1-q3*q3-q4*q4, 1*(q1*q3+q0*q4), 1*(q1*q4-q0*q3)}, {1*(q1*q3-q0*q4), q0*q0-q1*q1+q3*q3-q4*q4, 1*(q3*q4+q0*q1)}, {1*(q1*q4+q0*q3), 1*(q3*q4-q0*q1), q0*q0-q1*q1-q3*q3+q4*q4}});
}

/* This function returns a skew-symmetric matrix based on the values in vk
   Arguments:
     mat vk is a 3x1 vector
  Returns:
    mat skewMat a skew-symmetric 3x3 matrix of vk
*/
mat33 skewSymmetric(vec3 vk)
{
  double v1 = vk(0);
  double v2 = vk(1);
  double v3 = vk(2);
  //initializer_list<initializer_list<double>> skewSymexpected_contents = {{0.0, -v3, v2}, {v3, 0.0, -v1}, {-v2, v1, 0.0}};
  return mat({{0.0, -v3, v2}, {v3, 0.0, -v1}, {-v2, v1, 0.0}});
}



/*vector times a cube,summed along shared axis. Uses cubes third axis*/

mat vecTimesCube(vec v, cube c)
{
  if(v.n_elem!=c.n_slices)
  {
    throw new std::runtime_error("cube and vector don't align");
  }
  mat res = mat(arma::size(c.slice(0))).zeros();
  for(int i = 0; i < v.n_elem; i++)
  {
    res += v(i)*c.slice(i);
  }

  return res;
}

double sigmoid(double val)
{
  return 1.0/(1.0+exp(-val));
}

double swish(double val, double beta)
{
  return val*sigmoid(val*beta);
}


double shifted_swish(double val, double limit, double beta)
{
  return swish(val-limit,beta);
}

double softplus(double val, double k)
{
  double res = log(1.0+exp(val*k))/k;
  if(isinf(res)){ res = val;}
  return res;
}

double softplus_deriv(double val, double k)
{
  return 1.0/(1.0+exp(-val*k));
}

double softplus_deriv2(double val, double k)
{
  double evk = exp(val*k);
  // if(isinf(evk)){return 0;}
  return k*1.0/(2.0+evk+exp(-val*k));//pow(1+exp(val*k),2);

}

double shifted_softplus(double val, double limit, double k)
{
  return softplus(val-limit, k);
}

double shifted_softplus_deriv(double val, double limit, double k)
{
  return softplus_deriv(val-limit, k);
}

double shifted_softplus_deriv2(double val, double limit, double k)
{
  return softplus_deriv2(val-limit, k);
}

double smoothstep(double val){
  if(val<=0.0){
    return 0.0;
  }
  if(val>=1.0){
    return 1.0;
  }
  return 3.0*pow(val,2.0)-2.0*pow(val,3.0);
}

double smoothstep_deriv(double val){
  if(val<=0.0){
    return 0.0;
  }
  if(val>=1.0){
    return 0.0;
  }
  return 6.0*val-6.0*pow(val,2.0);
}

double smoothstep_deriv2(double val){
  if(val<=0.0){
    return 0.0;
  }
  if(val>=1.0){
    return 0.0;
  }
  return -6.0;
}

//
// double cost2AngOnly(vec4 q, vec3 v, vec3 u)
// {
//
//   q = normalise(q);
//   tuple<vec4,vec4> xy = quatSetBasis(v,u);
//   vec4 x = get<0>(xy);
//   vec4 y = get<1>(xy);
//   vec4 p = closestQuat(q,x,y);
//
//   double zz = abs(dot(q,p));
//   double sc = 2.0*acos(abs(zz));
//   return sc;
//
// }


/* This function finds the angle between a current and desired orientation, based on the current attitude quaternion, an ECI vector, and a body vector,
   where the desired orientation aligns the body vector and ECI vector. The quaternion is used to figure out where the body vector is relative to the ECI
   coordinate system.
   Arguments:
    u - 3 x 1 ECI vector
    v - 3 x 1 vector in body coordinates
    q - 4 x 1 attitude quaternion
    Returns:
     ang - angle between current quaternion and desired alignment
     dang - 3-axis representation of delta between angles
*/
// TODO Make static
tuple<double, vec4,mat44> cost2ang(vec4 q, vec3 s, vec3 e)
{
  double phi = 0.0;
  vec4 dphi = vec(4,fill::zeros);
  mat44 ddphi = mat(4,4,fill::zeros);
  if(e.is_zero())
  {
    return std::make_tuple(phi,dphi,ddphi);
  }

  q = normalise(q);
  s = normalise(s);
  e = normalise(e);

  mat::fixed<4,3> Wq = findWMat(q);

  double zz = dot(s,rotMat(q).t()*e);
  zz = min(max(zz,-1.0),1.0);
  double dphidz = 0.0;
  double ddphidzz = 0.0;
  if(abs(zz)<=acos_limitL){
    phi = acos(zz);
    dphidz = -1.0/sqrt(1.0-zz*zz);
    ddphidzz = -pow(1.0-zz*zz,-1.5)*zz;
  }else{
    double interp_valH = 0.0;
    double d_interp_valH = 0.0;
    double dd_interp_valH = 0.0;
    if(zz>0.0){
      double omz = 1.0-zz;
      interp_valH = sqrt(2.0)*pow(omz,0.5) + pow(omz,1.5)*(1.0/(6.0*sqrt(2.0))) + pow(omz,2.5)*(3.0/(80.0*sqrt(2.0))) + pow(omz,3.5)*(5.0/(448.0*sqrt(2.0)));
      d_interp_valH = -1.0*(0.5*sqrt(2.0)*pow(omz,-0.5) + pow(omz,0.5)*(1.5/(6.0*sqrt(2.0))) + pow(omz,1.5)*(3.0*2.5/(80.0*sqrt(2.0))) + pow(omz,2.5)*(5.0*3.5/(448.0*sqrt(2.0))));
      dd_interp_valH = (-0.5*0.5*sqrt(2.0)*pow(omz,-1.5) + pow(omz,-0.5)*(1.5*0.5/(6.0*sqrt(2.0))) + pow(omz,0.5)*(3.0*2.5*1.5/(80.0*sqrt(2.0))) + pow(omz,1.5)*(2.5*5.0*3.5/(448.0*sqrt(2.0))));
    }else{
      double opz = 1.0+zz;
      interp_valH = datum::pi-(sqrt(2.0)*pow(opz,0.5) + pow(opz,1.5)*(1.0/(6.0*sqrt(2.0))) + pow(opz,2.5)*(3.0/(80.0*sqrt(2.0))) + pow(opz,3.5)*(5.0/(448.0*sqrt(2.0))));
      d_interp_valH = -1.0*(0.5*sqrt(2.0)*pow(opz,-0.5) + pow(opz,0.5)*(1.5/(6.0*sqrt(2.0))) + pow(opz,1.5)*(3.0*2.5/(80.0*sqrt(2.0))) + pow(opz,2.5)*(5.0*3.5/(448.0*sqrt(2.0))));
      dd_interp_valH = -1.0*(-0.5*0.5*sqrt(2.0)*pow(opz,-1.5) + pow(opz,-0.5)*(1.5*0.5/(6.0*sqrt(2.0))) + pow(opz,0.5)*(3.0*2.5*1.5/(80.0*sqrt(2.0))) + pow(opz,1.5)*(2.5*5.0*3.5/(448.0*sqrt(2.0))));
    }

    if(abs(zz)<acos_limitH){
      double interp_range = acos_limitH-acos_limitL;
      double interp_dist = abs(zz)-acos_limitL;
      double interp_factorL = interp_dist/interp_range;
      double interp_factorH = 1.0-interp_factorL;

      double interp_valL = acos(zz);
      double d_interp_valL = -1.0/sqrt(1.0-zz*zz);
      double dd_interp_valL = -pow(1.0-zz*zz,-1.5)*zz;

      phi = interp_factorL*interp_valL + interp_factorH*interp_valH;
      dphidz = interp_factorL*d_interp_valL + interp_factorH*d_interp_valH;
      ddphidzz = interp_factorL*dd_interp_valL + interp_factorH*dd_interp_valH;
    }else{
      phi = interp_valH;
      dphidz = d_interp_valH;
      ddphidzz = dd_interp_valH;
    }
  }
  vec4 dzdq = (s.t()*dRTBdq(q,e)).t();
  mat44 ddzdqq = ddvTRTudq(q,s,e);
  dphi = dphidz*dzdq;
  ddphi = ddphidzz*dzdq*dzdq.t() + dphidz*ddzdqq;

  return std::make_tuple(phi,dphi,ddphi);


}


/* This function finds the angle between a current and desired orientation, based on the current attitude quaternion, an ECI vector, and a body vector,
   where the desired orientation aligns the body vector and ECI vector. The quaternion is used to figure out where the body vector is relative to the ECI
   coordinate system.
   Arguments:
    u - 3 x 1 ECI vector
    v - 3 x 1 vector in body coordinates
    q - 4 x 1 attitude quaternion
    Returns:
     ang - angle between current quaternion and desired alignment
     dang - 3-axis representation of delta between angles
*/
// TODO Make static
tuple<double, vec3,mat33> cost2angQ(vec4 q, vec3 s, vec3 e)
{
  double phi = 0.0;
  vec3 dphi = vec(3,fill::zeros);
  mat33 ddphi = mat(3,3,fill::zeros);
  if(e.is_zero())
  {
    return std::make_tuple(phi,dphi,ddphi);
  }

  q = normalise(q);
  s = normalise(s);
  e = normalise(e);

  mat::fixed<4,3> Wq = findWMat(q);
  double zz = norm_dot(s,rotMat(q).t()*e);
  // zz = min(max(zz,-1.0),1.0);
  double dphidz = 0.0;
  double ddphidzz = 0.0;
  if(abs(zz)<=acos_limitL){
    phi = acos(zz);
    dphidz = -1.0/sqrt(1.0-zz*zz);
    ddphidzz = -pow(1.0-zz*zz,-1.5)*zz;
  }else{
    double interp_valH = 0.0;
    double d_interp_valH = 0.0;
    double dd_interp_valH = 0.0;
    if(zz>0.0){
      double omz = 1.0-zz;

      interp_valH = sqrt(2.0)*pow(omz,0.5) + pow(omz,1.5)*(1.0/(6.0*sqrt(2.0))) + pow(omz,2.5)*(3.0/(80.0*sqrt(2.0))) + pow(omz,3.5)*(5.0/(448.0*sqrt(2.0)));
      if(abs(zz)>acos_limitH){omz = 1.0-acos_limitH;}
      d_interp_valH = -1.0*(0.5*sqrt(2.0)*pow(omz,-0.5) + pow(omz,0.5)*(1.5/(6.0*sqrt(2.0))) + pow(omz,1.5)*(3.0*2.5/(80.0*sqrt(2.0))) + pow(omz,2.5)*(5.0*3.5/(448.0*sqrt(2.0))));
      dd_interp_valH = (-0.5*0.5*sqrt(2.0)*pow(omz,-1.5) + pow(omz,-0.5)*(1.5*0.5/(6.0*sqrt(2.0))) + pow(omz,0.5)*(3.0*2.5*1.5/(80.0*sqrt(2.0))) + pow(omz,1.5)*(2.5*5.0*3.5/(448.0*sqrt(2.0))));
    }else{
      double opz = 1.0+zz;
      interp_valH = datum::pi-(sqrt(2.0)*pow(opz,0.5) + pow(opz,1.5)*(1.0/(6.0*sqrt(2.0))) + pow(opz,2.5)*(3.0/(80.0*sqrt(2.0))) + pow(opz,3.5)*(5.0/(448.0*sqrt(2.0))));
      if(abs(zz)>acos_limitH){opz = 1.0-acos_limitH;}
      d_interp_valH = -1.0*(0.5*sqrt(2.0)*pow(opz,-0.5) + pow(opz,0.5)*(1.5/(6.0*sqrt(2.0))) + pow(opz,1.5)*(3.0*2.5/(80.0*sqrt(2.0))) + pow(opz,2.5)*(5.0*3.5/(448.0*sqrt(2.0))));
      dd_interp_valH = -1.0*(-0.5*0.5*sqrt(2.0)*pow(opz,-1.5) + pow(opz,-0.5)*(1.5*0.5/(6.0*sqrt(2.0))) + pow(opz,0.5)*(3.0*2.5*1.5/(80.0*sqrt(2.0))) + pow(opz,1.5)*(2.5*5.0*3.5/(448.0*sqrt(2.0))));
    }

    if(abs(zz)<=acos_limitH){
      double interp_range = acos_limitH-acos_limitL;
      double interp_dist = abs(zz)-acos_limitL;
      double interp_factorL = interp_dist/interp_range;
      double interp_factorH = 1.0-interp_factorL;

      double interp_valL = acos(zz);
      double d_interp_valL = -1.0/sqrt(1.0-zz*zz);
      double dd_interp_valL = -pow(1.0-zz*zz,-1.5)*zz;

      phi = interp_factorL*interp_valL + interp_factorH*interp_valH;
      dphidz = interp_factorL*d_interp_valL + interp_factorH*d_interp_valH;
      ddphidzz = interp_factorL*dd_interp_valL + interp_factorH*dd_interp_valH;
    }else{
      phi = interp_valH;
      dphidz = d_interp_valH;
      ddphidzz = dd_interp_valH;
    }
  }
  vec3 dzdq = (s.t()*dRTBdqQ(q,e)).t();
  mat33 ddzdqq = ddvTRTudqQ(q,s,e);
  dphi = dphidz*dzdq;
  ddphi = ddphidzz*dzdq*dzdq.t() + dphidz*ddzdqq;

  return std::make_tuple(phi,dphi,ddphi);


}

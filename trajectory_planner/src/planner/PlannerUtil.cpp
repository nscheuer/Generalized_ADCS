#include "PlannerUtil.hpp"
#include "GeneralUtil.hpp"  // for normquaterr

using namespace arma;
using namespace std;


// ---- HELPERS ---

/* This function finds the angle between a current and desired orientation, based on the current attitude quaternion, an ECI vector, and a body vector,
   where the desired orientation aligns the body vector and ECI vector. The quaternion is used to figure out where the body vector is relative to the ECI
   coordinate system.
   Arguments:
    u - 3 x 1 ECI vector
    v - 3 x 1 vector in body coordinates
    q - 4 x 1 attitude quaternion
    Returns:
     ang - angle between current quaternion and desired alignment IN DEGREES
     dang - 3-axis representation of delta between angles IN DEGREES
*/

mat packageK(cube Kcube)
{
  mat K = mat(Kcube.n_rows*Kcube.n_cols, Kcube.n_slices).zeros();
  for(int k = 0; k < Kcube.n_slices; k++)
  {
    mat AqMatrix = Kcube.slice(k);
    for (size_t rowtest=0; rowtest < Kcube.n_rows; rowtest++)
    {
      for (size_t coltest=0; coltest < Kcube.n_cols; coltest++)
      {
        size_t i = rowtest*Kcube.n_cols+coltest;
        K(i, k) = AqMatrix(rowtest, coltest);
      }
    }
  }
  return K;
}

mat packageS(cube Scube)
  {
    mat S = mat(Scube.n_rows*Scube.n_cols, Scube.n_slices).zeros();
    for(int k = 0; k < Scube.n_slices; k++)
    {
      mat AqMatrix = Scube.slice(k);
      for (size_t rowtest=0; rowtest < Scube.n_rows; rowtest++)
      {
        for (size_t coltest=0; coltest < Scube.n_cols; coltest++)
        {
          size_t i = rowtest*Scube.n_cols+coltest;
          S(i, k) = AqMatrix(coltest, rowtest);
        }
      }
    }
    return S;
  }

VECTOR_INFO_FORM findVecTimes(VECTOR_INFO_FORM vecs_w_time,double dt,TIME_FORM time_start, TIME_FORM time_end){
  vec t = get<0>(vecs_w_time);
  mat r = get<1>(vecs_w_time);
  mat v = get<2>(vecs_w_time);
  mat b = get<3>(vecs_w_time);
  mat s = get<4>(vecs_w_time);
  mat a = get<5>(vecs_w_time);
  mat e = get<6>(vecs_w_time);
  vec p = get<7>(vecs_w_time);
  vec rho = get<8>(vecs_w_time);

  vec t0 = regspace(time_start,dt,time_end);
  vec ts;
  if(abs(time_end-t0.back())>EPSVAR){
    ts = unique(join_cols(t0,vec(1).ones()*time_end));
  }else{
    ts = unique(t0);
  }

  if((r.n_rows != 3) && (r.n_cols==3)){r = r.t();}
  if((v.n_rows != 3) && (v.n_cols==3)){v = v.t();}
  if((b.n_rows != 3) && (b.n_cols==3)){b = b.t();}
  if((s.n_rows != 3) && (s.n_cols==3)){s = s.t();}
  if((a.n_rows != 3) && (a.n_cols==3)){a = a.t();}
  if((e.n_rows != 3) && (e.n_cols==3)){e = e.t();}

  mat Rset = interp_vector(r,t,ts);//r.cols(inds_spaced);//timeAwareArma(r, dt, N, time_start, time_end);
  // mat dt_timevec = trans(extractRelevantTimes(r, dt, N, time_start, time_end));
  mat Vset = interp_vector(v,t,ts);//v.cols(inds_spaced);// trans(timeAwareArma(v, dt, N, time_start, time_end));
  mat Bset =  interp_vector(b,t,ts);//b.cols(inds_spaced);//trans(timeAwareArma(b, dt, N, time_start, time_end));
  mat sunset = interp_vector(s,t,ts);//s.cols(inds_spaced);// trans(timeAwareArma(s, dt, N, time_start, time_end));
  mat satvec = interp_vector(a,t,ts);//a.cols(inds_spaced);// trans(timeAwareArma(a, dt, N, time_start, time_end));
  mat ECIvec = interp_vector(e,t,ts);//e.cols(inds_spaced);// trans(timeAwareArma(e, dt, N, time_start, time_end));
  vec rhovec;
  interp1(t,rho,ts,rhovec);

  // vec prop_status = p(inds_spaced);
  vec prop_status = vec(ts.n_cols);
  interp1(t,p,ts,prop_status,"nearest");
  prop_status = round(prop_status);
  return std::make_tuple(ts,Rset,Vset,Bset,sunset,satvec,ECIvec,prop_status,rhovec);
}

mat interp_vector(mat m,vec t_of_m,vec t_des){
  mat output = mat(m.n_rows,t_des.n_elem).fill(datum::nan);
  vec tmp_data = vec(t_des.n_elem).fill(datum::nan);
  vec tmp_output = vec(t_des.n_elem).fill(datum::nan);
  for(int r = 0; r < m.n_rows; r++){
    tmp_data = m.row(r).t();
    interp1(t_of_m,tmp_data,t_des,tmp_output);
    output.row(r) = tmp_output.t();
  }
  return output;
}

// TRAJECTORY_FORM addTrajTimes(TRAJECTORY_FORM clean_traj ){
//   mat x = get<0>(clean_traj);
//   mat u = get<1>(clean_traj);
//   vec t = get<2>(clean_traj);
//   mat tq = get<3>(clean_traj);
//
//   mat x_times = addTimesToArma(x, t);
//   mat u_times = addTimesToArma(u, t);
//   mat tq_times = addTimesToArma(tq, t);
//   return std::make_tuple(x_times,u_times,t,tq_times);
// }



/*This function finds the Jacobian of the rk4z dynamics output
  Arguments:
    xk - state - 7 x 1 vector
    uk - control dipole vector - 3 x 1 vector
    tk - time - int
    dt0 - dt0 - double
  Outputs:
    7 x 7 matrix Ak the jacobian wrt x
    7 x 3 Bk the jacobian wrt u
*/
tuple<mat, mat,mat> rk4zJacobians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk;
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;

  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__dx0r = dxd2__dx2*dx2__dx2r*dx2r__dx0r;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;

  tuple<mat, mat, mat> jacK4 = sat.dynamicsJacobians(x3, uk, dynamics_info_kp1);
  mat dxd3__dx3 = get<0>(jacK4);
  mat dxd3__du = get<1>(jacK4);
  mat dxd3__dtorq = get<2>(jacK4);
  mat dx3__dx3r = sat.state_norm_jacobian(x3r);
  mat dx3r__dxd2 = dt0*I_state;
  mat dx3r__dx0 = I_state;
  mat dx3r__dx0r = dx0__dx0r + dx3r__dxd2*dxd2__dx0r;
  mat dx3r__du_ = dx3r__dxd2*dxd2__du_;
  mat dx3r__dtorq = dx3r__dxd2*dxd2__dtorq_;
  //mat E4 = get<1>(jacK4);
  mat dxd3__dx0 = dxd3__dx3*dx3__dx3r*dx3r__dx0;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__dx0r = dxd3__dx3*dx3__dx3r*dx3r__dx0r;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__du_ = dxd3__du + dxd3__dx3*dx3__dx3r*dx3r__du_;
  mat dxd3__dtorq_ = dxd3__dtorq + dxd3__dx3*dx3__dx3r*dx3r__dtorq;

  //Now get A, B
  //Ak = eye(length(xk)) + (dt0/6)*(m1+2*dxd1__dx0+2*dxd2__dx0+dxd3__dx0);
  //Bk = (dt0/6)*(n1 + 2*dxd1__du + 2*dxd2__du + dxd3__du);
  //mat eye = mat(sat.state_N(),7).eye();
  //mat Ak = mat(7,7).eye() + (dt0/6)*(dxd0__dx0+dxd1__dx0*2+dxd2__dx0*2+dxd3__dx0);
  //mat::fixed<7,3> Bk = (dt0/6)*(dxd0__du + dxd1__du*2 + dxd2__du*2 + dxd3__du);
  mat dxkp1__dxkp1r = sat.state_norm_jacobian(xkp1raw);
  return make_tuple(dxkp1__dxkp1r*(dx0__dx0r + (dt0/6.0)*(dxd0__dx0r+dxd1__dx0r*2.0+dxd2__dx0r*2.0+dxd3__dx0r)),
                    dxkp1__dxkp1r*((dt0/6.0)*(dxd0__du_ + dxd1__du_*2.0 + dxd2__du_*2.0 + dxd3__du_)),
                    dxkp1__dxkp1r*((dt0/6.0)*(dxd0__dtorq + dxd1__dtorq_*2.0 + dxd2__dtorq_*2.0 + dxd3__dtorq_)));
}

tuple<mat, mat,mat> rk4zx2Jacobians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk;
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;

  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;

  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;

  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  mat dx2__dtorq = dx2__dx2r*dx2r__dxd1*dxd1__dtorq_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  return make_tuple(dx2__dx0r,dx2__du_,dx2__dtorq);
}




tuple<cube, cube,cube> rk4zHessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__dx0r = dxd2__dx2*dx2__dx2r*dx2r__dx0r;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;r

  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));

  cube ddxd2__dx0rdx0r = matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)) + matOverCube(dxd2__dx2,ddx2__dx0rdx0r);

  cube ddxd2__du_dx0r = cubeTimesMat(ddxd2__dudx2,dx2__dx0r) + matOverCube(dxd2__dx2,ddx2__du_dx0r) +  matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r));
  cube ddxd2__du_du_ = ddxd2__dudu + matOverCube(dxd2__dx2,ddx2__du_du_)  + cubeTimesMat(ddxd2__dudx2,dx2__du_)+ matTimesCubeT(dx2__du_.t(),ddxd2__dudx2)  + matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__du_));

  tuple<mat, mat, mat> jacK4 = sat.dynamicsJacobians(x3, uk, dynamics_info_kp1);
  tuple<cube, cube, cube> hessK4 = sat.dynamicsHessians(x3, uk, dynamics_info_kp1);
  mat dxd3__dx3 = get<0>(jacK4);
  mat dxd3__du = get<1>(jacK4);
  mat dxd3__dtorq = get<2>(jacK4);
  cube ddxd3__dx3dx3 = get<0>(hessK4);
  cube ddxd3__dudx3 = get<1>(hessK4);
  cube ddxd3__dudu = get<2>(hessK4);
  mat dx3__dx3r = sat.state_norm_jacobian(x3r);
  cube ddx3__dx3rdx3r = sat.state_norm_hessian(x3r);
  mat dx3r__dxd2 = dt0*I_state;
  mat dx3r__dx0 = I_state;
  mat dx3r__dx0r = dx0__dx0r + dx3r__dxd2*dxd2__dx0r;
  mat dx3r__du_ = dx3r__dxd2*dxd2__du_;
  mat dx3r__dtorq = dx3r__dxd2*dxd2__dtorq_;
  mat dx3__dx0r = dx3__dx3r*dx3r__dx0r;
  mat dx3__du_ = dx3__dx3r*dx3r__dxd2*dxd2__du_;
  //mat E4 = get<1>(jacK4);

  mat dxd3__dx0 = dxd3__dx3*dx3__dx3r*dx3r__dx0;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__dx0r = dxd3__dx3*dx3__dx3r*dx3r__dx0r;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__du_ = dxd3__du + dxd3__dx3*dx3__dx3r*dx3r__du_;
  mat dxd3__dtorq_ = dxd3__dtorq + dxd3__dx3*dx3__dx3r*dx3r__dtorq;

  cube ddx3r__dx0rdx0r = matOverCube(dx3r__dx0,ddx0__dx0rdx0r) + matOverCube(dx3r__dxd2,ddxd2__dx0rdx0r);
  cube ddx3r__du_dx0r = matOverCube(dx3r__dxd2,ddxd2__du_dx0r);
  cube ddx3r__du_du_ =  matOverCube(dx3r__dxd2,ddxd2__du_du_);
  cube ddx3__du_du_ = matOverCube(dx3__dx3r,ddx3r__du_du_) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__du_));
  cube ddx3__du_dx0r = matOverCube(dx3__dx3r,ddx3r__du_dx0r) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));
  cube ddx3__dx0rdx0r = matOverCube(dx3__dx3r,ddx3r__dx0rdx0r) + matTimesCube(dx3r__dx0r.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));


  cube ddxd3__dx0rdx0r = matTimesCube(dx3__dx0r.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r)) + matOverCube(dxd3__dx3,ddx3__dx0rdx0r);
  cube ddxd3__du_dx0r = cubeTimesMat(ddxd3__dudx3,dx3__dx0r) + matOverCube(dxd3__dx3,ddx3__du_dx0r) +  matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r));
  cube ddxd3__du_du_ = ddxd3__dudu + matOverCube(dxd3__dx3,ddx3__du_du_)  + cubeTimesMat(ddxd3__dudx3,dx3__du_)+ matTimesCubeT(dx3__du_.t(),ddxd3__dudx3)  + matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__du_));


  //Now get A, B
  //Ak = eye(length(xk)) + (dt0/6)*(m1+2*dxd1__dx0+2*dxd2__dx0+dxd3__dx0);
  //Bk = (dt0/6)*(n1 + 2*dxd1__du + 2*dxd2__du + dxd3__du);
  //mat eye = mat(sat.state_N(),7).eye();
  //mat Ak = mat(7,7).eye() + (dt0/6)*(dxd0__dx0+dxd1__dx0*2+dxd2__dx0*2+dxd3__dx0);
  //mat::fixed<7,3> Bk = (dt0/6)*(dxd0__du + dxd1__du*2 + dxd2__du*2 + dxd3__du);
  mat dxkp1__dxkp1r = sat.state_norm_jacobian(xkp1raw);
  cube ddxkp1__dxkp1rdxkp1r = sat.state_norm_hessian(xkp1raw);

  mat dxkp1r__dxd0 = dt0*I_state*1.0/6.0;
  mat dxkp1r__dxd1 = dt0*I_state*2.0/6.0;
  mat dxkp1r__dxd2 = dt0*I_state*2.0/6.0;
  mat dxkp1r__dxd3 = dt0*I_state*1.0/6.0;
  mat dxkp1r__dx0 = I_state;
  mat dxkp1r__dx0r = dx0__dx0r + dxkp1r__dxd0*dxd0__dx0r + dxkp1r__dxd1*dxd1__dx0r + dxkp1r__dxd2*dxd2__dx0r + dxkp1r__dxd3*dxd3__dx0r;
  mat dxkp1r__du_ = dxkp1r__dxd0*dxd0__du_ + dxkp1r__dxd1*dxd1__du_ + dxkp1r__dxd2*dxd2__du_ + dxkp1r__dxd3*dxd3__du_;
  mat dxkp1__dx0r = dxkp1__dxkp1r*dxkp1r__dx0r;


  cube ddxkp1r__dx0rdx0r = matOverCube(dxkp1r__dx0,ddx0__dx0rdx0r) + matOverCube(dxkp1r__dxd0,ddxd0__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__dx0rdx0r);
  cube ddxkp1r__du_dx0r =  matOverCube(dxkp1r__dxd0,ddxd0__du_dx0r) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__du_dx0r) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__du_dx0r) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__du_dx0r);
  cube ddxkp1r__du_du_ =   matOverCube(dxkp1r__dxd0,ddxd0__du_du_) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__du_du_) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__du_du_) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__du_du_);


  cube ddxkp1__dx0rdx0r = matOverCube(dxkp1__dxkp1r,ddxkp1r__dx0rdx0r) + matTimesCube(dxkp1r__dx0r.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__dx0r));
  cube ddxkp1__du_dx0r =  matOverCube(dxkp1__dxkp1r,ddxkp1r__du_dx0r) + matTimesCube(dxkp1r__du_.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__dx0r));
  cube ddxkp1__du_du_ =  matOverCube(dxkp1__dxkp1r,ddxkp1r__du_du_) + matTimesCube(dxkp1r__du_.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__du_));

  return make_tuple(ddxkp1__dx0rdx0r,ddxkp1__du_dx0r,ddxkp1__du_du_);
}


tuple<cube, cube,cube,mat,mat,cube,cube,cube,cube,cube> rk4zxkp1rHessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__dx0r = dxd2__dx2*dx2__dx2r*dx2r__dx0r;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;r

  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));

  cube ddxd2__dx0rdx0r = matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)) + matOverCube(dxd2__dx2,ddx2__dx0rdx0r);

  cube ddxd2__du_dx0r = cubeTimesMat(ddxd2__dudx2,dx2__dx0r) + matOverCube(dxd2__dx2,ddx2__du_dx0r) +  matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r));
  cube ddxd2__du_du_ = ddxd2__dudu + matOverCube(dxd2__dx2,ddx2__du_du_)  + cubeTimesMat(ddxd2__dudx2,dx2__du_)+ matTimesCubeT(dx2__du_.t(),ddxd2__dudx2)  + matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__du_));

  tuple<mat, mat, mat> jacK4 = sat.dynamicsJacobians(x3, uk, dynamics_info_kp1);
  tuple<cube, cube, cube> hessK4 = sat.dynamicsHessians(x3, uk, dynamics_info_kp1);
  mat dxd3__dx3 = get<0>(jacK4);
  mat dxd3__du = get<1>(jacK4);
  mat dxd3__dtorq = get<2>(jacK4);
  cube ddxd3__dx3dx3 = get<0>(hessK4);
  cube ddxd3__dudx3 = get<1>(hessK4);
  cube ddxd3__dudu = get<2>(hessK4);
  mat dx3__dx3r = sat.state_norm_jacobian(x3r);
  cube ddx3__dx3rdx3r = sat.state_norm_hessian(x3r);
  mat dx3r__dxd2 = dt0*I_state;
  mat dx3r__dx0 = I_state;
  mat dx3r__dx0r = dx0__dx0r + dx3r__dxd2*dxd2__dx0r;
  mat dx3r__du_ = dx3r__dxd2*dxd2__du_;
  mat dx3r__dtorq = dx3r__dxd2*dxd2__dtorq_;
  mat dx3__dx0r = dx3__dx3r*dx3r__dx0r;
  mat dx3__du_ = dx3__dx3r*dx3r__dxd2*dxd2__du_;
  //mat E4 = get<1>(jacK4);

  mat dxd3__dx0 = dxd3__dx3*dx3__dx3r*dx3r__dx0;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__dx0r = dxd3__dx3*dx3__dx3r*dx3r__dx0r;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__du_ = dxd3__du + dxd3__dx3*dx3__dx3r*dx3r__du_;
  mat dxd3__dtorq_ = dxd3__dtorq + dxd3__dx3*dx3__dx3r*dx3r__dtorq;

  cube ddx3r__dx0rdx0r = matOverCube(dx3r__dx0,ddx0__dx0rdx0r) + matOverCube(dx3r__dxd2,ddxd2__dx0rdx0r);
  cube ddx3r__du_dx0r = matOverCube(dx3r__dxd2,ddxd2__du_dx0r);
  cube ddx3r__du_du_ =  matOverCube(dx3r__dxd2,ddxd2__du_du_);
  cube ddx3__du_du_ = matOverCube(dx3__dx3r,ddx3r__du_du_) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__du_));
  cube ddx3__du_dx0r = matOverCube(dx3__dx3r,ddx3r__du_dx0r) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));
  cube ddx3__dx0rdx0r = matOverCube(dx3__dx3r,ddx3r__dx0rdx0r) + matTimesCube(dx3r__dx0r.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));


  cube ddxd3__dx0rdx0r = matTimesCube(dx3__dx0r.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r)) + matOverCube(dxd3__dx3,ddx3__dx0rdx0r);
  cube ddxd3__du_dx0r = cubeTimesMat(ddxd3__dudx3,dx3__dx0r) + matOverCube(dxd3__dx3,ddx3__du_dx0r) +  matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r));
  cube ddxd3__du_du_ = ddxd3__dudu + matOverCube(dxd3__dx3,ddx3__du_du_)  + cubeTimesMat(ddxd3__dudx3,dx3__du_)+ matTimesCubeT(dx3__du_.t(),ddxd3__dudx3)  + matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__du_));


  //Now get A, B
  //Ak = eye(length(xk)) + (dt0/6)*(m1+2*dxd1__dx0+2*dxd2__dx0+dxd3__dx0);
  //Bk = (dt0/6)*(n1 + 2*dxd1__du + 2*dxd2__du + dxd3__du);
  //mat eye = mat(sat.state_N(),7).eye();
  //mat Ak = mat(7,7).eye() + (dt0/6)*(dxd0__dx0+dxd1__dx0*2+dxd2__dx0*2+dxd3__dx0);
  //mat::fixed<7,3> Bk = (dt0/6)*(dxd0__du + dxd1__du*2 + dxd2__du*2 + dxd3__du);
  mat dxkp1__dxkp1r = sat.state_norm_jacobian(xkp1raw);
  cube ddxkp1__dxkp1rdxkp1r = sat.state_norm_hessian(xkp1raw);

  mat dxkp1r__dxd0 = dt0*I_state*1.0/6.0;
  mat dxkp1r__dxd1 = dt0*I_state*2.0/6.0;
  mat dxkp1r__dxd2 = dt0*I_state*2.0/6.0;
  mat dxkp1r__dxd3 = dt0*I_state*1.0/6.0;
  mat dxkp1r__dx0 = I_state;
  mat dxkp1r__dx0r = dx0__dx0r + dxkp1r__dxd0*dxd0__dx0r + dxkp1r__dxd1*dxd1__dx0r + dxkp1r__dxd2*dxd2__dx0r + dxkp1r__dxd3*dxd3__dx0r;
  mat dxkp1r__du_ = dxkp1r__dxd0*dxd0__du_ + dxkp1r__dxd1*dxd1__du_ + dxkp1r__dxd2*dxd2__du_ + dxkp1r__dxd3*dxd3__du_;
  mat dxkp1__dx0r = dxkp1__dxkp1r*dxkp1r__dx0r;



  cube ddxkp1r__dx0rdx0r = matOverCube(dxkp1r__dx0,ddx0__dx0rdx0r) + matOverCube(dxkp1r__dxd0,ddxd0__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__dx0rdx0r) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__dx0rdx0r);
  cube ddxkp1r__du_dx0r =  matOverCube(dxkp1r__dxd0,ddxd0__du_dx0r) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__du_dx0r) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__du_dx0r) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__du_dx0r);
  cube ddxkp1r__du_du_ =   matOverCube(dxkp1r__dxd0,ddxd0__du_du_) \
                          + matOverCube(dxkp1r__dxd1,ddxd1__du_du_) \
                          + matOverCube(dxkp1r__dxd2,ddxd2__du_du_) \
                          + matOverCube(dxkp1r__dxd3,ddxd3__du_du_);


  cube ddxkp1__dx0rdx0r = matOverCube(dxkp1__dxkp1r,ddxkp1r__dx0rdx0r) + matTimesCube(dxkp1r__dx0r.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__dx0r));
  cube ddxkp1__du_dx0r =  matOverCube(dxkp1__dxkp1r,ddxkp1r__du_dx0r) + matTimesCube(dxkp1r__du_.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__dx0r));
  cube ddxkp1__du_du_ =  matOverCube(dxkp1__dxkp1r,ddxkp1r__du_du_) + matTimesCube(dxkp1r__du_.t(),cubeTimesMat(ddxkp1__dxkp1rdxkp1r,dxkp1r__du_));

  return make_tuple(ddxkp1r__dx0rdx0r,ddxkp1r__du_dx0r,ddxkp1r__du_du_,dxkp1r__dx0r,dxkp1r__du_,ddx0__dx0rdx0r,ddxd0__dx0rdx0r,ddxd1__dx0rdx0r,ddxd2__dx0rdx0r,ddxd3__dx0rdx0r);
}




tuple<cube, cube,cube,mat,mat> rk4zx3rHessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians


  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__dx0r = dxd2__dx2*dx2__dx2r*dx2r__dx0r;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));

  cube ddxd2__dx0rdx0r = matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)) + matOverCube(dxd2__dx2,ddx2__dx0rdx0r);

  cube ddxd2__du_dx0r = cubeTimesMat(ddxd2__dudx2,dx2__dx0r) + matOverCube(dxd2__dx2,ddx2__du_dx0r) +  matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r));
  cube ddxd2__du_du_ = ddxd2__dudu + matOverCube(dxd2__dx2,ddx2__du_du_)  + cubeTimesMat(ddxd2__dudx2,dx2__du_)+ matTimesCubeT(dx2__du_.t(),ddxd2__dudx2)  + matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__du_));

  tuple<mat, mat, mat> jacK4 = sat.dynamicsJacobians(x3, uk, dynamics_info_kp1);
  tuple<cube, cube, cube> hessK4 = sat.dynamicsHessians(x3, uk, dynamics_info_kp1);
  mat dxd3__dx3 = get<0>(jacK4);
  mat dxd3__du = get<1>(jacK4);
  mat dxd3__dtorq = get<2>(jacK4);
  cube ddxd3__dx3dx3 = get<0>(hessK4);
  cube ddxd3__dudx3 = get<1>(hessK4);
  cube ddxd3__dudu = get<2>(hessK4);
  mat dx3__dx3r = sat.state_norm_jacobian(x3r);
  cube ddx3__dx3rdx3r = sat.state_norm_hessian(x3r);
  mat dx3r__dxd2 = dt0*I_state;
  mat dx3r__dx0 = I_state;
  mat dx3r__dx0r = dx0__dx0r + dx3r__dxd2*dxd2__dx0r;
  mat dx3r__du_ = dx3r__dxd2*dxd2__du_;
  mat dx3r__dtorq = dx3r__dxd2*dxd2__dtorq_;
  mat dx3__dx0r = dx3__dx3r*dx3r__dx0r;
  mat dx3__du_ = dx3__dx3r*dx3r__dxd2*dxd2__du_;
  //mat E4 = get<1>(jacK4);
  mat dxd3__dx0 = dxd3__dx3*dx3__dx3r*dx3r__dx0;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__du_ = dxd3__du + dxd3__dx3*dx3__dx3r*dx3r__du_;
  mat dxd3__dtorq_ = dxd3__dtorq + dxd3__dx3*dx3__dx3r*dx3r__dtorq;

  cube ddx3r__dx0rdx0r = matOverCube(dx3r__dx0,ddx0__dx0rdx0r) + matOverCube(dx3r__dxd2,ddxd2__dx0rdx0r);
  cube ddx3r__du_dx0r = matOverCube(dx3r__dxd2,ddxd2__du_dx0r);
  cube ddx3r__du_du_ =  matOverCube(dx3r__dxd2,ddxd2__du_du_);
  cube ddx3__du_du_ = matOverCube(dx3__dx3r,ddx3r__du_du_) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__du_));
  cube ddx3__du_dx0r = matOverCube(dx3__dx3r,ddx3r__du_dx0r) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));
  cube ddx3__dx0rdx0r = matOverCube(dx3__dx3r,ddx3r__dx0rdx0r) + matTimesCube(dx3r__dx0r.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));

  // cube ddxd3__dx0rdx0r = matTimesCube(dx3__dx0r.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r)) + matOverCube(dxd3__dx3,ddx3__dx0rdx0r);
  // cube ddxd3__du_dx0r = cubeTimesMat(ddxd3__dudx3,dx3__dx0r) + matOverCube(dxd3__dx3,ddx3__du_dx0r) +  matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r));
  // cube ddxd3__du_du_ = ddxd3__dudu + matOverCube(dxd3__dx3,ddx3__du_du_)  + 2.0*cubeTimesMat(ddxd3__dudx3,dx3__du_) + matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__du_));


  return make_tuple(ddx3r__dx0rdx0r,ddx3r__du_dx0r,ddx3r__du_du_,dx3r__dx0r,dx3r__du_);
}





tuple<cube, cube,cube> rk4zxd2Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));

  cube ddxd2__dx0rdx0r = matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)) + matOverCube(dxd2__dx2,ddx2__dx0rdx0r);
  cout<<"hesstest xd2\n";
  cout<<matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)).slice(3)<<"\n";
  cout<<matOverCube(dxd2__dx2,ddx2__dx0rdx0r).slice(3)<<"\n";
  cout<<ddxd2__dx0rdx0r.slice(3)<<"\n";
  cube ddxd2__du_dx0r = cubeTimesMat(ddxd2__dudx2,dx2__dx0r) + matOverCube(dxd2__dx2,ddx2__du_dx0r) +  matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r));
  cube ddxd2__du_du_ = ddxd2__dudu + matOverCube(dxd2__dx2,ddx2__du_du_)  + cubeTimesMat(ddxd2__dudx2,dx2__du_)+ matTimesCubeT(dx2__du_.t(),ddxd2__dudx2)  + matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__du_));

  // cube ddxd3__dx0rdx0r = matTimesCube(dx3__dx0r.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r)) + matOverCube(dxd3__dx3,ddx3__dx0rdx0r);
  // cube ddxd3__du_dx0r = cubeTimesMat(ddxd3__dudx3,dx3__dx0r) + matOverCube(dxd3__dx3,ddx3__du_dx0r) +  matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r));
  // cube ddxd3__du_du_ = ddxd3__dudu + matOverCube(dxd3__dx3,ddx3__du_du_)  + 2.0*cubeTimesMat(ddxd3__dudx3,dx3__du_) + matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__du_));


  return make_tuple(ddxd2__dx0rdx0r,ddxd2__du_dx0r,ddxd2__du_du_);
}


tuple<cube, cube,cube> rk4zx3Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r.t().t();
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r.t().t();
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r.t().t();
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1.t().t();
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians



  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__dx0r = dxd2__dx2*dx2__dx2r*dx2r__dx0r;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;r

  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));

  cube ddxd2__dx0rdx0r = matTimesCube(dx2__dx0r.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r)) + matOverCube(dxd2__dx2,ddx2__dx0rdx0r);

  cube ddxd2__du_dx0r = cubeTimesMat(ddxd2__dudx2,dx2__dx0r) + matOverCube(dxd2__dx2,ddx2__du_dx0r) +  matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__dx0r));
  cube ddxd2__du_du_ = ddxd2__dudu + matOverCube(dxd2__dx2,ddx2__du_du_)  + cubeTimesMat(ddxd2__dudx2,dx2__du_)+ matTimesCubeT(dx2__du_.t(),ddxd2__dudx2)  + matTimesCube(dx2__du_.t(),cubeTimesMat(ddxd2__dx2dx2,dx2__du_));

  tuple<mat, mat, mat> jacK4 = sat.dynamicsJacobians(x3, uk, dynamics_info_kp1);
  tuple<cube, cube, cube> hessK4 = sat.dynamicsHessians(x3, uk, dynamics_info_kp1);
  mat dxd3__dx3 = get<0>(jacK4);
  mat dxd3__du = get<1>(jacK4);
  mat dxd3__dtorq = get<2>(jacK4);
  cube ddxd3__dx3dx3 = get<0>(hessK4);
  cube ddxd3__dudx3 = get<1>(hessK4);
  cube ddxd3__dudu = get<2>(hessK4);
  mat dx3__dx3r = sat.state_norm_jacobian(x3r);
  cube ddx3__dx3rdx3r = sat.state_norm_hessian(x3r);
  mat dx3r__dxd2 = dt0*I_state;
  mat dx3r__dx0 = I_state;
  mat dx3r__dx0r = dx0__dx0r + dx3r__dxd2*dxd2__dx0r;
  mat dx3r__du_ = dx3r__dxd2*dxd2__du_;
  mat dx3r__dtorq = dx3r__dxd2*dxd2__dtorq_;
  mat dx3__dx0r = dx3__dx3r*dx3r__dx0r;
  mat dx3__du_ = dx3__dx3r*dx3r__dxd2*dxd2__du_;
  //mat E4 = get<1>(jacK4);
  mat dxd3__dx0 = dxd3__dx3*dx3__dx3r*dx3r__dx0;//(dxd3__dx3 + dt0*dxd3__dx3*dxd2__dx0);
  mat dxd3__du_ = dxd3__du + dxd3__dx3*dx3__dx3r*dx3r__du_;
  mat dxd3__dtorq_ = dxd3__dtorq + dxd3__dx3*dx3__dx3r*dx3r__dtorq;

  cube ddx3r__dx0rdx0r = matOverCube(dx3r__dx0,ddx0__dx0rdx0r) + matOverCube(dx3r__dxd2,ddxd2__dx0rdx0r);
  cube ddx3r__du_dx0r = matOverCube(dx3r__dxd2,ddxd2__du_dx0r);
  cube ddx3r__du_du_ =  matOverCube(dx3r__dxd2,ddxd2__du_du_);
  cube ddx3__du_du_ = matOverCube(dx3__dx3r,ddx3r__du_du_) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__du_));
  cube ddx3__du_dx0r = matOverCube(dx3__dx3r,ddx3r__du_dx0r) + matTimesCube(dx3r__du_.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));
  cube ddx3__dx0rdx0r = matOverCube(dx3__dx3r,ddx3r__dx0rdx0r) + matTimesCube(dx3r__dx0r.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r));
  cout<<"hesstest x3\n";
  cout<<matOverCube(dx3__dx3r,ddx3r__dx0rdx0r).slice(3)<<"\n";
  cout<<matTimesCube(dx3r__dx0r.t(),cubeTimesMat(ddx3__dx3rdx3r,dx3r__dx0r)).slice(3)<<"\n";

  // cube ddxd3__dx0rdx0r = matTimesCube(dx3__dx0r.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r)) + matOverCube(dxd3__dx3,ddx3__dx0rdx0r);
  // cube ddxd3__du_dx0r = cubeTimesMat(ddxd3__dudx3,dx3__dx0r) + matOverCube(dxd3__dx3,ddx3__du_dx0r) +  matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__dx0r));
  // cube ddxd3__du_du_ = ddxd3__dudu + matOverCube(dxd3__dx3,ddx3__du_du_)  + 2.0*cubeTimesMat(ddxd3__dudx3,dx3__du_) + matTimesCube(dx3__du_.t(),cubeTimesMat(ddxd3__dx3dx3,dx3__du_));


  return make_tuple(ddx3__dx0rdx0r,ddx3__du_dx0r,ddx3__du_du_);
}


tuple<cube, cube,cube,mat,mat> rk4zx1Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0r_dx0r_ = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);

  cube ddxd0__dx0r_dx0r_ = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0r_dx0r_);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);

  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;// + dx1r__dxd0*dxd0__dx0;
  mat dx1r__dx0r_ = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r_;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;

  // .mat dx1__dx0r = dx1__dx1r*dx1r__dx0r_;
  // .mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddx1r__dx0r_dx0r_ = matOverCube(dx1r__dx0,ddx0__dx0r_dx0r_) + matOverCube(dx1r__dxd0,ddxd0__dx0r_dx0r_);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);

  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r_));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0r_dx0r_) + matTimesCube(dx1r__dx0r_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r_));

  return make_tuple(ddx1__dx0rdx0r,ddx1__du_dx0r,ddx1__du_du_,dx1__dx0r,dx1__du_);
}


tuple<cube, cube,cube> rk4zxd0Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  //mat E2 = get<1>(jacK2);
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);

  return make_tuple(ddxd0__dx0rdx0r,ddxd0__du_dx0r,ddxd0__du_du_);
}


tuple<cube, cube,cube> rk4zxd1Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;

  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));

  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__du_dx0r);
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_)+ matTimesCubeT(dx1__du_.t(),ddxd1__dudx1)  + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  return make_tuple(ddxd1__dx0rdx0r,ddxd1__du_dx0r,ddxd1__du_du_);
}

tuple<cube, cube,cube,mat,mat> rk4zx2rHessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_)+ matTimesCubeT(dx1__du_.t(),ddxd1__dudx1)  + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));
  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  return make_tuple(ddx2r__dx0rdx0r,ddx2r__du_dx0r,ddx2r__du_du_,dx2r__dx0r,dx2r__du_);
}

tuple<cube, cube,cube,mat,mat> rk4zx2Hessians(double dt0,vec xk, vec uk,const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );
  vec xkraw = xk.t().t();
  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1r = xk+xd0*0.5*dt0;
  vec x1 = x1r;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk, dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2r = xk+xd1*0.5*dt0;
  vec x2 = x2r;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk,  dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3r = xk+xd2*dt0;
  vec x3 = x3r;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec xkp1 = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  vec xkp1raw = xkp1;
  xkp1 = sat.state_norm(xkp1);
  // xkp1(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(xkp1(span(sat.quat0index(),sat.quat0index()+3)));
  mat Gk = sat.findGMat(xk.rows(sat.quat0index(),sat.quat0index()+3));
  mat G2 = sat.findGMat(x1r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G3 = sat.findGMat(x2r.rows(sat.quat0index(),sat.quat0index()+3));
  mat G4 = sat.findGMat(x3r.rows(sat.quat0index(),sat.quat0index()+3));
  mat Gkp1 = sat.findGMat(xkp1.rows(sat.quat0index(),sat.quat0index()+3));
  //Now for the dynamics Jacobians

  mat I_state = mat(sat.state_N(),sat.state_N()).eye();

  //mat33 skewSymU = 2*sat.invJcom*skewSymmetric(uk);//*sat.invJcom;
  mat dx0__dx0r = sat.state_norm_jacobian(xkraw);
  cube ddx0__dx0rdx0r = sat.state_norm_hessian(xkraw);
  tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
  tuple<cube, cube, cube> hessK1 = sat.dynamicsHessians(xk, uk, dynamics_info_k);
  mat dxd0__dx0 = get<0>(jacK1);
  mat dxd0__du_ = get<1>(jacK1);
  mat dxd0__dtorq = get<2>(jacK1);
  mat dxd0__dx0r = dxd0__dx0*dx0__dx0r;

  cube ddxd0__dx0dx0 = get<0>(hessK1);
  cube ddxd0__du_dx0 = get<1>(hessK1);
  cube ddxd0__du_du_ = get<2>(hessK1);
  tuple<mat, mat, mat> jacK2 = sat.dynamicsJacobians(x1, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK2 = sat.dynamicsHessians(x1, uk, dynamics_info_mid);
  mat dxd1__dx1 = get<0>(jacK2);
  mat dxd1__du = get<1>(jacK2);
  mat dxd1__dtorq = get<2>(jacK2);
  cube ddxd1__dx1dx1 = get<0>(hessK2);
  cube ddxd1__dudx1 = get<1>(hessK2);
  cube ddxd1__dudu = get<2>(hessK2);
  //mat E2 = get<1>(jacK2);
  mat dx1__dx1r = sat.state_norm_jacobian(x1r);
  cube ddx1__dx1rdx1r = sat.state_norm_hessian(x1r);
  mat dx1r__dxd0 = 0.5*dt0*I_state;
  mat dx1r__dx0 = I_state;
  mat dx1r__dx0r = dx0__dx0r + dx1r__dxd0*dxd0__dx0r;
  mat dx1r__du_ = dx1r__dxd0*dxd0__du_;
  mat dx1r__dtorq = dx1r__dxd0*dxd0__dtorq;
  mat dx1__du_ = dx1__dx1r*dx1r__dxd0*dxd0__du_;


  mat dx1__dx0r = dx1__dx1r*dx1r__dx0r;
  mat dxd1__dx0 = dxd1__dx1*dx1__dx1r*dx1r__dx0;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;
  mat dxd1__dx0r = dxd1__dx1*dx1__dx1r*dx1r__dx0r;//dxd1__dx1 + 0.5*dt0*dxd1__dx1*dxd0__dx0;


  mat dxd1__du_ = dxd1__du + dxd1__dx1*dx1__dx1r*dx1r__du_;
  mat dxd1__dtorq_ = dxd1__dtorq + dxd1__dx1*dx1__dx1r*dx1r__dtorq;
  cube ddxd0__dx0rdx0r = matTimesCube(dx0__dx0r.t(),cubeTimesMat(ddxd0__dx0dx0,dx0__dx0r)) + matOverCube(dxd0__dx0,ddx0__dx0rdx0r);
  cube ddxd0__du_dx0r = cubeTimesMat(ddxd0__du_dx0,dx0__dx0r);
  cube ddx1r__dx0rdx0r = matOverCube(dx1r__dx0,ddx0__dx0rdx0r) + matOverCube(dx1r__dxd0,ddxd0__dx0rdx0r);
  cube ddx1r__du_dx0r = matOverCube(dx1r__dxd0,ddxd0__du_dx0r);
  cube ddx1r__du_du_ =  matOverCube(dx1r__dxd0,ddxd0__du_du_);
  cube ddx1__du_du_ = matOverCube(dx1__dx1r,ddx1r__du_du_) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__du_));
  cube ddx1__du_dx0r = matOverCube(dx1__dx1r,ddx1r__du_dx0r) + matTimesCube(dx1r__du_.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));
  cube ddx1__dx0rdx0r = matOverCube(dx1__dx1r,ddx1r__dx0rdx0r) + matTimesCube(dx1r__dx0r.t(),cubeTimesMat(ddx1__dx1rdx1r,dx1r__dx0r));


  tuple<mat, mat, mat> jacK3 = sat.dynamicsJacobians(x2, uk, dynamics_info_mid);
  tuple<cube, cube, cube> hessK3 = sat.dynamicsHessians(x2, uk, dynamics_info_mid);
  mat dxd2__dx2 = get<0>(jacK3);
  mat dxd2__du = get<1>(jacK3);
  mat dxd2__dtorq = get<2>(jacK3);

  cube ddxd2__dx2dx2 = get<0>(hessK3);
  cube ddxd2__dudx2 = get<1>(hessK3);
  cube ddxd2__dudu = get<2>(hessK3);
  mat dx2__dx2r = sat.state_norm_jacobian(x2r);
  cube ddx2__dx2rdx2r = sat.state_norm_hessian(x2r);
  mat dx2r__dxd1 = 0.5*dt0*I_state;
  mat dx2r__dx0 = I_state;
  mat dx2r__dx0r = dx0__dx0r + dx2r__dxd1*dxd1__dx0r;
  mat dx2r__du_ = dx2r__dxd1*dxd1__du_;
  mat dx2r__dtorq = dx2r__dxd1*dxd1__dtorq_;
  mat dx2__dx0r = dx2__dx2r*dx2r__dx0r;
  mat dx2__du_ = dx2__dx2r*dx2r__dxd1*dxd1__du_;
  //mat E3 = get<1>(jacK3);
  mat dxd2__dx0 = dxd2__dx2*dx2__dx2r*dx2r__dx0;//dxd2__dx2 + 0.5*dt0*dxd2__dx2*dxd1__dx0;
  mat dxd2__du_ = dxd2__du + dxd2__dx2*dx2__dx2r*dx2r__du_;
  mat dxd2__dtorq_ = dxd2__dtorq + dxd2__dx2*dx2__dx2r*dx2r__dtorq;
  cube ddxd1__dx0rdx0r = matTimesCube(dx1__dx0r.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r)) + matOverCube(dxd1__dx1,ddx1__dx0rdx0r);
  cube ddxd1__du_dx0r = cubeTimesMat(ddxd1__dudx1,dx1__dx0r) + matOverCube(dxd1__dx1,ddx1__du_dx0r) +  matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__dx0r));
  cube ddxd1__du_du_ = ddxd1__dudu + matOverCube(dxd1__dx1,ddx1__du_du_) + cubeTimesMat(ddxd1__dudx1,dx1__du_) + matTimesCubeT(dx1__du_.t(),ddxd1__dudx1) + matTimesCube(dx1__du_.t(),cubeTimesMat(ddxd1__dx1dx1,dx1__du_));
  cube ddx2r__dx0rdx0r = matOverCube(dx2r__dx0,ddx0__dx0rdx0r) + matOverCube(dx2r__dxd1,ddxd1__dx0rdx0r);
  cube ddx2r__du_dx0r = matOverCube(dx2r__dxd1,ddxd1__du_dx0r);
  cube ddx2r__du_du_ =  matOverCube(dx2r__dxd1,ddxd1__du_du_);
  cube ddx2__du_du_ = matOverCube(dx2__dx2r,ddx2r__du_du_) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__du_));

  cube ddx2__du_dx0r = matOverCube(dx2__dx2r,ddx2r__du_dx0r) + matTimesCube(dx2r__du_.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  cube ddx2__dx0rdx0r = matOverCube(dx2__dx2r,ddx2r__dx0rdx0r) + matTimesCube(dx2r__dx0r.t(),cubeTimesMat(ddx2__dx2rdx2r,dx2r__dx0r));
  tuple<cube,cube,cube,mat,mat> testres = rk4zx2rHessians( dt0, xk,  uk, sat,  dynamics_info_k,  dynamics_info_kp1);

  if(!approx_equal(ddx2r__du_du_,get<2>(testres),"absdiff",1e-09)){
    cout<<"not equal!\n";
    cout<<ddx2r__du_du_<<"\n";
    cout<<get<2>(testres)<<"\n";
    throw new std::runtime_error("ddx2r__du_du_ not equal.");
  }

  if(!approx_equal(dx2r__du_,get<4>(testres),"absdiff",1e-09)){
    cout<<"not equal!\n";
    cout<<dx2r__du_<<"\n";
    cout<<get<4>(testres)<<"\n";
    throw new std::runtime_error("dx2r__du_ not equal.");
  }
  return make_tuple(ddx2__dx0rdx0r,ddx2__du_dx0r,ddx2__du_du_,dx2__dx0r,dx2__du_);
}


/*This function propagates the dynamics forward using rk4
  Arguments:
    xk - 7 x 1 state vector
    uk - 3 x 1 control dipole vector
    tk - int time
    dt - double dt
    Bk - 3 x 1 magnetic field vector
  Returns:
   dynamics integrated forward by 1 timestep using rk4 - 7 x 1 vector
   hi
*/
vec rk4z_pure(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  tuple<vec,vec> rk4zout = rk4z(dt0,xk,uk,sat,dynamics_info_k,dynamics_info_kp1);
  vec out = get<0>(rk4zout);
  return out;
}


tuple<vec,vec> rk4z(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec dist_torq0 = get<1>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec out = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  out = sat.state_norm(out);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return make_tuple(out,dist_torq0);
}



vec rk4zxkp1r(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  vec out = (xk + (dt0/6.0)*(xd0+xd1*2.0+xd2*2.0+xd3));
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return out;
}


vec rk4zxd3(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  x3 = sat.state_norm(x3);
  // x3(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x3(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x3, uk, dynamics_info_kp1);
  vec xd3 = get<0>(dynout);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return xd3;
}

vec rk4zx1(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return x1;
}


vec rk4zxd0(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return xd0;
}

vec rk4zx2(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return x2;
}


vec rk4zx2r(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return x2;
}
vec rk4zxd1(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return xd1;
}


vec rk4zx3(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  x3 = sat.state_norm(x3);
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return x3;
}



vec rk4zx3r(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return x3;
}



vec rk4zxd2(double dt0, vec xk, vec uk, const Satellite& sat, const DYNAMICS_INFO_FORM& dynamics_info_k, const DYNAMICS_INFO_FORM& dynamics_info_kp1)
{
  //Now, get different dynamics outputs xd0....xd3
  DYNAMICS_INFO_FORM dynamics_info_mid = make_tuple(
                          0.5*(get<0>(dynamics_info_k)+get<0>(dynamics_info_kp1)),
                          0.5*(get<1>(dynamics_info_k)+get<1>(dynamics_info_kp1)),
                          get<2>(dynamics_info_k)*get<2>(dynamics_info_kp1),
                          0.5*(get<3>(dynamics_info_k)+get<3>(dynamics_info_kp1)),
                          0.5*(get<4>(dynamics_info_k)+get<4>(dynamics_info_kp1)),
                          get<5>(dynamics_info_k)*get<5>(dynamics_info_kp1)
                          );

  xk = sat.state_norm(xk);
  tuple<vec,vec> dynout = sat.dynamics(xk, uk, dynamics_info_k);
  vec xd0 = get<0>(dynout);
  vec x1 = xk+xd0*0.5*dt0;
  x1 = sat.state_norm(x1);
  // x1(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x1(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x1, uk,dynamics_info_mid);
  vec xd1 = get<0>(dynout);
  vec x2 = xk+xd1*0.5*dt0;
  x2 = sat.state_norm(x2);
  // x2(span(sat.quat0index(),sat.quat0index()+3)) = normalise(x2(span(sat.quat0index(),sat.quat0index()+3)));
  dynout = sat.dynamics(x2, uk, dynamics_info_mid);
  vec xd2 = get<0>(dynout);
  vec x3 = xk+xd2*dt0;
  // out(span(sat.quat0index(),sat.quat0index()+3)) =  normalise(out(span(sat.quat0index(),sat.quat0index()+3)));
  return xd2;
}

vec4 qdes(vec3 satvk, vec3 ECIvk, vec4 q, vec3 w, vec3 Bbody, mat33 wt)
{
    satvk = normalise(satvk);
    ECIvk = normalise(ECIvk);
    double x0 = 0.5*norm(satvk + ECIvk);
    vec3 xv = cross(satvk,ECIvk)/(2.0*x0);
    vec3 yv = (satvk+ECIvk)/(2.0*x0);
    //vec3 a = normalise(cross(Bbody,cross(Bbody,wt*w)));
    vec3 qv = q.rows(1,3);
    double q0 = as_scalar(q.row(0));

    vec3 xqv = -q0*xv + x0*qv - cross(xv,qv);
    double xq0 = x0*q0 - dot(-xv,qv);
    vec3 yqv = -q0*yv - cross(yv,qv);
    double yq0 = -dot(-yv,qv);

    double bcdx = dot(normalise(wt*Bbody),xqv);
    double bcdy = dot(normalise(wt*Bbody),yqv);

    double gmag = pow(pow(bcdx,2.0)+pow(bcdy,2.0),0.5);
    double g0 = bcdy*xq0/gmag;
    //vec3 gv = (bcdy*xqv - bcdx*yqv)/gmag;
    vec3 gv = normalise(cross(normalise(cross(yqv,xqv)),normalise(wt*Bbody)));

    // vec g0v(1,fill::none);
    // g0v.fill(g0);
    return normalise(join_cols(vec({g0}),gv));

}

REG_PAIR increaseReg(REG_PAIR reg0, REG_SETTINGS_FORM regSettings_tmp){
    double regMin_tmp = get<1>(regSettings_tmp);
    double regMax_tmp = get<2>(regSettings_tmp);
    double regScale_tmp = get<3>(regSettings_tmp);

    double rho0 = get<0>(reg0);
    double drho0 = get<1>(reg0);

    double drho = max(drho0*regScale_tmp,regScale_tmp);
    double rho = max(rho0*drho,regMin_tmp);

    // Cap regularization at maximum to prevent overflow
    // When rho hits max, stop increasing drho to avoid numerical issues
    if(rho >= regMax_tmp){
      rho = regMax_tmp;
      drho = drho0;  // Don't keep increasing the scale factor
    }

    if(isinf(rho)){
      cout<<"rho should not be inf (increase reg failure)\n";
      throw("rho should not be inf");
    }
    return make_tuple(rho,drho);
}

REG_PAIR decreaseReg(REG_PAIR reg0, REG_SETTINGS_FORM regSettings_tmp){
    double regMin_tmp = get<1>(regSettings_tmp);
    double regMax_tmp = get<2>(regSettings_tmp);
    double regScale_tmp = get<3>(regSettings_tmp);
    int regMinConds = get<5>(regSettings_tmp);

    double rho0 = get<0>(reg0);
    double drho0 = get<1>(reg0);

    double drho = min(drho0/regScale_tmp,1.0/regScale_tmp);
    double rho = drho*rho0;
    switch(regMinConds){
      case 0:
        break;
      case 1:
        rho = max(rho,regMin_tmp);
        break;
      case 2:
        rho = rho*(rho>regMin_tmp);
        break;
      default:
        throw("invalid regmin conditions");
    }
    rho = min(rho,regMax_tmp);
    if(isinf(rho)){
      cout<<"rho should not be inf (decrease reg failure)\n";
      throw("rho should not be inf");
    }
    return make_tuple(rho,drho);
}

// ============================================================================
// K-GAIN WARM-START FUNCTIONS
// ============================================================================

/**
 * Unpackage flattened K-gains back to 3D cube
 * Inverse of packageK()
 */
cube unpackageK(const mat& Kflat, int n_ctrl, int n_reduced) {
    int N = Kflat.n_cols;
    cube Kcube(n_ctrl, n_reduced, N, fill::zeros);

    for (int k = 0; k < N; k++) {
        for (int row = 0; row < n_ctrl; row++) {
            for (int col = 0; col < n_reduced; col++) {
                int flat_idx = row * n_reduced + col;
                Kcube(row, col, k) = Kflat(flat_idx, k);
            }
        }
    }
    return Kcube;
}

/**
 * Convert quaternion error to 3-vector representation
 * @param qerr Normalized quaternion error (q_ref^{-1} * q_current)
 * @param mode 0=MRP+, 1=MRP, 2=Cayley (default), 3=qv+, 4=qv
 */
vec3 quatTo3Vec(const vec4& qerr, int mode) {
    vec4 q = qerr;

    // Ensure positive scalar for modes 0, 3
    if (mode == 0 || mode == 3) {
        if (q(0) < 0) q = -q;
    }

    switch (mode) {
        case 0:  // MRP with positive scalar
        case 1:  // MRP
            return 2.0 * q.rows(1, 3) / (1.0 + q(0));

        case 2:  // Cayley parameters (default, matches planner)
            if (abs(q(0)) < EPSVAR) {
                q(0) = (q(0) >= 0) ? EPSVAR : -EPSVAR;
            }
            return q.rows(1, 3) / q(0);

        case 3:  // Vector part with positive scalar
        case 4:  // Vector part
            return q.rows(1, 3);

        default:
            throw runtime_error("quatTo3Vec: unsupported mode");
    }
}

/**
 * SLERP-interpolate a trajectory to finer timestep
 * Linear interpolation for omega and h_rw, SLERP for quaternion
 */
mat slerpInterpolateTrajectory(const mat& Xset_coarse, double dt_coarse, double dt_fine, double tf) {
    int N_coarse = Xset_coarse.n_cols;
    int N_fine = int(tf / dt_fine) + 1;
    int n_state = Xset_coarse.n_rows;

    mat Xset_fine(n_state, N_fine, fill::zeros);

    // Extract quaternions and ensure continuity (no sign flips)
    mat quats_coarse = Xset_coarse.rows(3, 6);  // 4 x N_coarse
    for (int k = 1; k < N_coarse; k++) {
        if (dot(quats_coarse.col(k), quats_coarse.col(k-1)) < 0) {
            quats_coarse.col(k) *= -1;
        }
    }

    // Compute cubic spline coefficients for non-quaternion states
    // For each row, compute spline coefficients using natural spline boundary conditions
    vec t_coarse = linspace(0, tf, N_coarse);
    vec t_fine_vec = linspace(0, tf, N_fine);
    
    // Helper lambda for cubic spline interpolation
    // Uses natural spline (second derivative = 0 at endpoints)
    auto cubicSplineInterpolate = [&](const rowvec& y_coarse) -> rowvec {
        int n = N_coarse;
        rowvec y_fine(N_fine);
        
        // Compute spline coefficients using tridiagonal system
        vec h(n-1);  // intervals
        for (int i = 0; i < n-1; i++) {
            h(i) = t_coarse(i+1) - t_coarse(i);
        }
        
        // Set up tridiagonal system for second derivatives (natural spline)
        vec alpha(n-1);
        for (int i = 1; i < n-1; i++) {
            alpha(i) = 3.0/h(i) * (y_coarse(i+1) - y_coarse(i)) - 
                       3.0/h(i-1) * (y_coarse(i) - y_coarse(i-1));
        }
        
        vec l(n), mu(n), z(n);
        l(0) = 1.0; mu(0) = 0.0; z(0) = 0.0;
        
        for (int i = 1; i < n-1; i++) {
            l(i) = 2.0 * (t_coarse(i+1) - t_coarse(i-1)) - h(i-1) * mu(i-1);
            mu(i) = h(i) / l(i);
            z(i) = (alpha(i) - h(i-1) * z(i-1)) / l(i);
        }
        
        l(n-1) = 1.0; z(n-1) = 0.0;
        
        vec c(n), b(n-1), d(n-1);
        c(n-1) = 0.0;
        
        for (int j = n-2; j >= 0; j--) {
            c(j) = z(j) - mu(j) * c(j+1);
            b(j) = (y_coarse(j+1) - y_coarse(j)) / h(j) - h(j) * (c(j+1) + 2.0*c(j)) / 3.0;
            d(j) = (c(j+1) - c(j)) / (3.0 * h(j));
        }
        
        // Evaluate spline at fine grid points
        for (int k = 0; k < N_fine; k++) {
            double t = t_fine_vec(k);
            
            // Find interval
            int i = 0;
            for (int j = 0; j < n-1; j++) {
                if (t >= t_coarse(j) && t <= t_coarse(j+1)) {
                    i = j;
                    break;
                }
                if (j == n-2) i = n-2;  // clamp to last interval
            }
            
            double dt_local = t - t_coarse(i);
            y_fine(k) = y_coarse(i) + b(i)*dt_local + c(i)*dt_local*dt_local + d(i)*dt_local*dt_local*dt_local;
        }
        
        return y_fine;
    };

    // Cubic interpolation for angular velocity (rows 0-2)
    for (int row = 0; row < 3; row++) {
        Xset_fine.row(row) = cubicSplineInterpolate(Xset_coarse.row(row));
    }

    // Cubic interpolation for RW momentum (rows 7+)
    for (int row = 7; row < n_state; row++) {
        Xset_fine.row(row) = cubicSplineInterpolate(Xset_coarse.row(row));
    }

    // SLERP for quaternions (keep existing approach but use fine grid times)
    for (int k = 0; k < N_fine; k++) {
        double t = k * dt_fine;

        // Find bounding coarse indices
        double idx_float = t / dt_coarse;
        int i0 = min(int(floor(idx_float)), N_coarse - 2);
        int i1 = i0 + 1;
        double alpha = idx_float - i0;

        // SLERP for quaternion (using LERP + normalize as approximation)
        vec4 q0 = normalise(quats_coarse.col(i0));
        vec4 q1 = normalise(quats_coarse.col(i1));

        // Ensure same hemisphere
        if (dot(q0, q1) < 0) {
            q1 = -q1;
        }

        // LERP + normalize (good approximation for small intervals)
        vec4 q_interp = normalise((1.0 - alpha) * q0 + alpha * q1);
        Xset_fine.submat(3, k, 6, k) = q_interp;
    }

    return Xset_fine;
}

/**
 * Generate warm-start trajectory for fine grid using K-gain feedback
 *
 * Propagates dynamics at fine timestep while applying K-gain corrections from
 * coarse optimization. This preserves local optimality while adapting to finer
 * discretization through closed-loop simulation.
 */
TRAJECTORY_FORM kgainWarmStart(
    const mat& Xset_coarse,
    const mat& Uset_coarse,
    const mat& Kset_flat,
    double dt_coarse,
    double dt_fine,
    double tf,
    Satellite& sat,
    VECTOR_INFO_FORM& vecs_fine,
    int quat_to_3vec_mode
) {
    int N_coarse = Xset_coarse.n_cols;
    int N_fine = int(tf / dt_fine) + 1;
    int n_state = sat.state_N();         // 7 + n_rw
    int n_ctrl = sat.control_N();        // n_mtq + n_rw
    int n_reduced = sat.reduced_state_N(); // 6 + n_rw
    int n_rw = sat.number_RW;
    int n_mtq = sat.number_MTQ;

    // 1. SLERP-interpolate reference trajectory to fine grid
    mat Xset_ref = slerpInterpolateTrajectory(Xset_coarse, dt_coarse, dt_fine, tf);

    // Ensure correct length
    if ((int)Xset_ref.n_cols < N_fine) {
        mat pad = repmat(Xset_ref.col(Xset_ref.n_cols - 1), 1, N_fine - Xset_ref.n_cols);
        Xset_ref = join_rows(Xset_ref, pad);
    } else if ((int)Xset_ref.n_cols > N_fine) {
        Xset_ref = Xset_ref.head_cols(N_fine);
    }

    // 2. Unpack K-gains from flattened format
    int N_K = Kset_flat.n_cols;
    cube Kset = unpackageK(Kset_flat, n_ctrl, n_reduced);

    // Pad Kset if needed
    if (N_K < N_coarse - 1) {
        cube pad(n_ctrl, n_reduced, (N_coarse - 1) - N_K, fill::zeros);
        for (int i = 0; i < (int)pad.n_slices; i++) {
            pad.slice(i) = Kset.slice(N_K - 1);
        }
        Kset = join_slices(Kset, pad);
    }

    // 3. Scale RW controls to physical units for feedback computation
    // Pass1 returns Uset in optimizer units for RW, Kset in optimizer units
    // K-gains map physical state errors (rad/s, rad) to optimizer-unit control corrections
    // We compute: u_phys = u_nom_phys + du_phys, where du_phys = du_opt * NONMTQ_TORQ_SCALE
    mat Uset_phys = Uset_coarse;
    if (n_rw > 0) {
        // Scale RW controls to physical: u_phys = u_opt * NONMTQ_TORQ_SCALE
        Uset_phys.rows(n_mtq, n_ctrl - 1) *= NONMTQ_TORQ_SCALE;
    }
    // K-gains stay in optimizer units - they output optimizer-unit corrections

    // 4. Initialize output arrays
    mat Xset_fine(n_state, N_fine, fill::zeros);
    mat Uset_fine(n_ctrl, N_fine, fill::zeros);
    mat TQset_fine(3, N_fine, fill::zeros);
    vec t_fine = linspace(0, tf, N_fine);

    // Extract dynamics info from vecs_fine
    mat Bset = get<3>(vecs_fine);  // B-field (3, N_fine)
    mat Rset = get<1>(vecs_fine);  // Position (3, N_fine)
    mat Vset = get<2>(vecs_fine);  // Velocity (3, N_fine)
    mat Sset = get<4>(vecs_fine);  // Sun vector (3, N_fine)
    vec pset = get<7>(vecs_fine);  // Eclipse flag (N_fine)

    // 5. Initialize from first state
    vec x_sim = Xset_coarse.col(0);
    x_sim = sat.state_norm(x_sim);
    Xset_fine.col(0) = x_sim;

    // 6. Propagate with K-gain feedback
    for (int k = 0; k < N_fine - 1; k++) {
        // Find coarse index (ZOH - use K at start of interval)
        int i = min(int(k * dt_fine / dt_coarse), N_coarse - 2);
        i = max(i, 0);
        i = min(i, (int)Kset.n_slices - 1);

        // Get nominal control (ZOH) in physical units
        int u_idx = min(i, (int)Uset_phys.n_cols - 1);
        vec u_nom = Uset_phys.col(u_idx);

        // Get K-gain at this coarse index (no interpolation)
        mat K = Kset.slice(i);

        // Compute reduced error state: dx = x_sim - x_ref
        vec x_ref = Xset_ref.col(k);
        vec dx(n_reduced, fill::zeros);

        // 6a. Angular velocity error
        dx.head(3) = x_sim.head(3) - x_ref.head(3);

        // 6b. Quaternion error -> 3-vector
        vec4 q_ref = normalise(x_ref.rows(3, 6));
        vec4 q_sim = normalise(x_sim.rows(3, 6));
        vec4 q_err = normquaterr(q_ref, q_sim);
        dx.rows(3, 5) = quatTo3Vec(q_err, quat_to_3vec_mode);

        // 6c. RW momentum error (if applicable)
        if (n_rw > 0) {
            dx.rows(6, n_reduced - 1) = x_sim.tail(n_rw) - x_ref.tail(n_rw);
        }

        // Apply K-gain feedback correction with timestep ratio scaling
        double dt_ratio = dt_fine / dt_coarse;
        vec du = K * dx * dt_ratio;
        if (n_rw > 0) {
            // Scale RW correction from optimizer to physical units
            du.rows(n_mtq, n_ctrl - 1) *= NONMTQ_TORQ_SCALE;
        }
        vec u = u_nom + du;  // Both in physical units now

        Uset_fine.col(k) = u;

        // Create dynamics info for this timestep
        int k_next = min(k + 1, N_fine - 1);
        DYNAMICS_INFO_FORM dyn_k = make_tuple(
            vec3(Bset.col(k)), vec3(Rset.col(k)), (int)pset(k),
            vec3(Vset.col(k)), vec3(Sset.col(k)), 0);
        DYNAMICS_INFO_FORM dyn_kp1 = make_tuple(
            vec3(Bset.col(k_next)), vec3(Rset.col(k_next)), (int)pset(k_next),
            vec3(Vset.col(k_next)), vec3(Sset.col(k_next)), 0);

        // Convert control back to optimizer units for C++ dynamics
        // C++ dynamics internally scales RW by NONMTQ_TORQ_SCALE
        vec u_opt = u;
        if (n_rw > 0) {
            u_opt.rows(n_mtq, n_ctrl - 1) /= NONMTQ_TORQ_SCALE;
        }
        
        // Propagate dynamics using rk4z (expects optimizer units)
        tuple<vec, vec> dynout = rk4z(dt_fine, x_sim, u_opt, sat, dyn_k, dyn_kp1);
        x_sim = sat.state_norm(get<0>(dynout));
        Xset_fine.col(k + 1) = x_sim;
        TQset_fine.col(k) = get<1>(dynout);

        // Early exit on NaN
        if (x_sim.has_nan()) {
            break;
        }
    }

    // Copy final control
    Uset_fine.col(N_fine - 1) = Uset_fine.col(N_fine - 2);

    return make_tuple(Xset_fine, Uset_fine, t_fine, TQset_fine);
}

// --- Closed-loop inverse dynamics warm-start ---
// At each fine timestep, computes controls from the ACTUAL simulated state
// (not the reference), targeting the next reference angular velocity.
// This prevents drift compounding that breaks open-loop warm-starts.
TRAJECTORY_FORM closedLoopInvDynWarmStart(
    const mat& Xset_coarse,
    double dt_coarse,
    double dt_fine,
    double tf,
    Satellite& sat,
    VECTOR_INFO_FORM& vecs_fine
) {
    int N_fine = int(tf / dt_fine) + 1;
    int n_state = sat.state_N();
    int n_ctrl = sat.control_N();
    int n_rw = sat.number_RW;
    int n_mtq = sat.number_MTQ;

    // 1. SLERP-interpolate reference trajectory to fine grid
    mat Xset_ref = slerpInterpolateTrajectory(Xset_coarse, dt_coarse, dt_fine, tf);
    if ((int)Xset_ref.n_cols < N_fine) {
        mat pad = repmat(Xset_ref.col(Xset_ref.n_cols - 1), 1, N_fine - Xset_ref.n_cols);
        Xset_ref = join_rows(Xset_ref, pad);
    } else if ((int)Xset_ref.n_cols > N_fine) {
        Xset_ref = Xset_ref.head_cols(N_fine);
    }

    // 2. Extract environment data
    mat Bset = get<3>(vecs_fine);
    mat Rset = get<1>(vecs_fine);
    mat Vset = get<2>(vecs_fine);
    mat Sset = get<4>(vecs_fine);
    vec pset = get<7>(vecs_fine);

    // 3. Get satellite properties
    mat33 J = sat.Jcom_noRW;

    // Get actuator limits
    double m_max = 1.0;
    if (n_mtq > 0) {
        m_max = sat.MTQ_max[0];
    }
    double rw_torq_max = 0.0;
    if (n_rw > 0) {
        rw_torq_max = sat.RW_max_torq[0];
    }

    // Angular acceleration limit (rad/s²) — prevents huge torques from bad reference
    double w_dot_max = 0.1;

    // 4. Initialize output arrays
    mat Xset_fine(n_state, N_fine, fill::zeros);
    mat Uset_fine(n_ctrl, N_fine, fill::zeros);
    mat TQset_fine(3, N_fine, fill::zeros);
    vec t_fine = linspace(0, tf, N_fine);

    // 5. Closed-loop propagation
    vec x_sim = Xset_coarse.col(0);
    x_sim = sat.state_norm(x_sim);
    Xset_fine.col(0) = x_sim;

    for (int k = 0; k < N_fine - 1; k++) {
        vec3 w_sim = x_sim.head(3);
        vec4 q_sim = normalise(x_sim.rows(3, 6));

        // Target: reference ω at next timestep
        vec3 w_target = Xset_ref(span(0, 2), span(k + 1, k + 1));
        vec3 w_dot_desired = (w_target - w_sim) / dt_fine;

        // Rate-limit angular acceleration
        double w_dot_mag = norm(w_dot_desired);
        if (w_dot_mag > w_dot_max) {
            w_dot_desired *= (w_dot_max / w_dot_mag);
        }

        // Required torque from Euler equation: τ = J*ω̇ + ω×(J*ω)
        vec3 tau_needed = J * w_dot_desired + cross(w_sim, J * w_sim);

        // RW: project needed torque onto RW axes
        vec3 tau_rw = zeros<vec>(3);
        if (n_rw > 0) {
            for (int i = 0; i < n_rw; i++) {
                vec3 ax = sat.RW_axes[i];
                double rw_torque = dot(tau_needed, ax);
                rw_torque = clamp(rw_torque, -rw_torq_max, rw_torq_max);
                Uset_fine(n_mtq + i, k) = rw_torque;
                tau_rw += rw_torque * ax;
            }
        }

        // MTQ: solve for dipole from actual body-frame B
        vec3 tau_mtq_needed = tau_needed - tau_rw;
        mat33 R = rotMat(q_sim);
        int k_idx = min(k, (int)Bset.n_cols - 1);
        vec3 B_body = R.t() * vec3(Bset.col(k_idx));
        double B_sq = dot(B_body, B_body);

        vec3 m_dipole = zeros<vec>(3);
        if (B_sq > 1e-20) {
            m_dipole = cross(B_body, tau_mtq_needed) / B_sq;
            // Check for NaN/inf
            if (m_dipole.has_nan() || !m_dipole.is_finite()) {
                m_dipole.zeros();
            }
            m_dipole = clamp(m_dipole, -m_max, m_max);
        }
        for (int i = 0; i < n_mtq && i < 3; i++) {
            Uset_fine(i, k) = m_dipole(i);
        }

        // Convert controls to optimizer units for rk4z
        vec u_opt = Uset_fine.col(k);
        if (n_rw > 0) {
            u_opt.rows(n_mtq, n_ctrl - 1) /= NONMTQ_TORQ_SCALE;
        }

        // Build dynamics info
        int k_next = min(k + 1, N_fine - 1);
        DYNAMICS_INFO_FORM dyn_k = make_tuple(
            vec3(Bset.col(k)), vec3(Rset.col(k)), (int)pset(k),
            vec3(Vset.col(k)), vec3(Sset.col(k)), 0);
        DYNAMICS_INFO_FORM dyn_kp1 = make_tuple(
            vec3(Bset.col(k_next)), vec3(Rset.col(k_next)), (int)pset(k_next),
            vec3(Vset.col(k_next)), vec3(Sset.col(k_next)), 0);

        // Propagate one step
        tuple<vec, vec> dynout = rk4z(dt_fine, x_sim, u_opt, sat, dyn_k, dyn_kp1);
        x_sim = sat.state_norm(get<0>(dynout));
        Xset_fine.col(k + 1) = x_sim;
        TQset_fine.col(k) = get<1>(dynout);

        // Early exit on NaN
        if (x_sim.has_nan()) {
            cout << "WARNING: closedLoopInvDynWarmStart NaN at k=" << k << ", aborting\n";
            break;
        }
    }

    // Copy final control
    if (N_fine > 1) {
        Uset_fine.col(N_fine - 1) = Uset_fine.col(N_fine - 2);
    }

    return make_tuple(Xset_fine, Uset_fine, t_fine, TQset_fine);
}

// --- END HELPERS ---

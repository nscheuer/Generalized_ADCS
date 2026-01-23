#include "Satellite.hpp"
#include <stdexcept>
#include <list>
#include <vector>

namespace py = pybind11;
using namespace arma;
using namespace std;




static const double cost_hess_mult = 1.0;



Satellite::Satellite() {
  Jcom = mat33().eye();
  invJcom = mat33().eye();
  plan_for_aero = 0;
  plan_for_prop = 0;
  plan_for_srp = 0;
  plan_for_gg = 0;
  plan_for_resdipole = 0;
  plan_for_gendist = 0;
  coeff_N = 1e8;
}
Satellite::Satellite(mat33 Jcom_in) {
  Jcom = Jcom_in;
  invJcom = Jcom_in.i();
  update_invJ_noRW();
  plan_for_aero = 0;
  plan_for_prop = 0;
  plan_for_srp = 0;
  plan_for_gg = 0;
  plan_for_resdipole = 0;
  plan_for_gendist = 0;
  coeff_N = 1e8;
}

void Satellite::change_Jcom(mat33 Jcom_in) {
  Jcom = Jcom_in;
  invJcom = Jcom_in.i();
  update_invJ_noRW();
}
void Satellite::change_Jcom_py(py::array_t<double> Jcom_in_py) {
  mat33 Jcom_in = numpyToArmaMatrix(Jcom_in_py);
  Jcom = Jcom_in;
  invJcom = inv(Jcom);
  update_invJ_noRW();
}
void Satellite::readJcom(){
  cout<<Jcom<<"\n";
}



int Satellite::constraint_N() const {return int(useAVconstraint) + number_sunpoints + 2*number_MTQ + 2*number_magic + (2+2+1)*number_RW;}//all are inequalities in this setup
int Satellite::ineq_constraint_N() const {return constraint_N();}
int Satellite::control_N() const {return number_MTQ + number_magic + number_RW;}
int Satellite::state_N() const {return 7+number_RW;}
int Satellite::reduced_state_N() const {return state_N() - 1;}
int Satellite::quat0index() const {return 3;}
int Satellite::redang0index() const {return 3;}
int Satellite::avindex0() const {return 0;}
int Satellite::eq_constraint_N() const {return 0;}// constraint_N() - ineq_constraint_N();}

std::tuple<double> Satellite::constraintSettingsTuple(){return make_tuple(0);}



void Satellite::add_gg_torq(){
  plan_for_gg = 1;
}
void Satellite::remove_gg_torq(){
  plan_for_gg = 0;
}

void Satellite::add_prop_torq(vec3 prop_torq_in){
  plan_for_prop = 1;
  prop_torq = prop_torq_in;
}
void Satellite::add_prop_torq_py(py::array_t<double> prop_torq_in_py){
  vec3 prop_torq_in = numpyToArmaVector(prop_torq_in_py);
  plan_for_prop = 1;
  prop_torq = prop_torq_in;
}
void Satellite::remove_prop_torq(){
  plan_for_prop = 0;
  prop_torq *= 0;
}

void Satellite::add_srp_torq(arma::mat srp_coeff_in,int coeff_len){
  plan_for_srp = 1;
  srp_coeff *= 0;
  srp_coeff = srp_coeff_in;
  coeff_N = min(coeff_len,coeff_N);
}
void Satellite::add_srp_torq_py(py::array_t<double> srp_coeff_in_py,int coeff_len){
  mat srp_coeff_in = numpyToArmaMatrix(srp_coeff_in_py);
  plan_for_srp = 1;
  srp_coeff *= 0;
  srp_coeff = srp_coeff_in;
  coeff_N = min(coeff_len,coeff_N);
}
void Satellite::remove_srp_torq(){
  plan_for_srp = 0;
  srp_coeff *= 0;
}

void Satellite::add_aero_torq(arma::mat drag_coeff_in,int coeff_len ){
  plan_for_aero = 1;
  drag_coeff *= 0;
  drag_coeff = drag_coeff_in;
  coeff_N = min(coeff_len,coeff_N);
}
void Satellite::add_aero_torq_py(py::array_t<double> drag_coeff_in_py,int coeff_len){
  mat drag_coeff_in = numpyToArmaMatrix(drag_coeff_in_py);
  plan_for_aero = 1;
  drag_coeff *= 0;
  drag_coeff = drag_coeff_in;
  coeff_N = min(coeff_len,coeff_N);
}
void Satellite::remove_aero_torq(){
  plan_for_aero = 0;
  drag_coeff *= 0;
}

// Surface-based SRP disturbance methods (Option B)
void Satellite::set_srp_surfaces(mat normals, mat centroids, vec areas,
                                  vec eta_s, vec eta_d, vec eta_a, vec3 COM){
  // Ensure matrices are 3 x N
  if(normals.n_rows != 3 && normals.n_cols == 3){ normals = normals.t(); }
  if(centroids.n_rows != 3 && centroids.n_cols == 3){ centroids = centroids.t(); }

  num_srp_surfaces = normals.n_cols;
  srp_normals = normals;
  srp_centroids = centroids;
  srp_areas = areas;
  srp_eta_s = eta_s;
  srp_eta_d = eta_d;
  srp_eta_a = eta_a;
  srp_COM = COM;
  plan_for_srp = 1;
}

void Satellite::set_srp_surfaces_py(py::array_t<double> normals_py, py::array_t<double> centroids_py,
                                     py::array_t<double> areas_py, py::array_t<double> eta_s_py,
                                     py::array_t<double> eta_d_py, py::array_t<double> eta_a_py,
                                     py::array_t<double> COM_py){
  mat normals = numpyToArmaMatrix(normals_py);
  mat centroids = numpyToArmaMatrix(centroids_py);
  vec areas = numpyToArmaVector(areas_py);
  vec eta_s = numpyToArmaVector(eta_s_py);
  vec eta_d = numpyToArmaVector(eta_d_py);
  vec eta_a = numpyToArmaVector(eta_a_py);
  vec3 COM = numpyToArmaVector(COM_py);
  set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);
}

void Satellite::clear_srp_surfaces(){
  plan_for_srp = 0;
  num_srp_surfaces = 0;
  srp_normals.reset();
  srp_centroids.reset();
  srp_areas.reset();
  srp_eta_s.reset();
  srp_eta_d.reset();
  srp_eta_a.reset();
  srp_COM.zeros();
}

// Surface-based aerodynamic drag disturbance methods (Option B)
void Satellite::set_drag_surfaces(mat normals, mat centroids, vec areas,
                                   vec CDs, vec3 COM){
  // Ensure matrices are 3 x N
  if(normals.n_rows != 3 && normals.n_cols == 3){ normals = normals.t(); }
  if(centroids.n_rows != 3 && centroids.n_cols == 3){ centroids = centroids.t(); }

  num_drag_surfaces = normals.n_cols;
  drag_normals = normals;
  drag_centroids = centroids;
  drag_areas = areas;
  drag_CDs = CDs;
  drag_COM = COM;
  plan_for_aero = 1;
}

void Satellite::set_drag_surfaces_py(py::array_t<double> normals_py, py::array_t<double> centroids_py,
                                      py::array_t<double> areas_py, py::array_t<double> CDs_py,
                                      py::array_t<double> COM_py){
  mat normals = numpyToArmaMatrix(normals_py);
  mat centroids = numpyToArmaMatrix(centroids_py);
  vec areas = numpyToArmaVector(areas_py);
  vec CDs = numpyToArmaVector(CDs_py);
  vec3 COM = numpyToArmaVector(COM_py);
  set_drag_surfaces(normals, centroids, areas, CDs, COM);
}

void Satellite::clear_drag_surfaces(){
  plan_for_aero = 0;
  num_drag_surfaces = 0;
  drag_normals.reset();
  drag_centroids.reset();
  drag_areas.reset();
  drag_CDs.reset();
  drag_COM.zeros();
}

void Satellite::add_resdipole_torq(vec3 rd_in){
  plan_for_resdipole = 1;
  res_dipole = rd_in;
}
void Satellite::add_resdipole_torq_py(py::array_t<double> rd_in_py){
  vec3 rd_in = numpyToArmaVector(rd_in_py);
  plan_for_resdipole = 1;
  res_dipole = rd_in;
}
void Satellite::remove_resdipole_torq(){
  plan_for_resdipole = 0;
  res_dipole *= 0;
}

void Satellite::add_gendist_torq(vec3 gd_torq_in){
  plan_for_gendist = 1;
  gen_dist_torq = gd_torq_in;
}
void Satellite::add_gendist_torq_py(py::array_t<double> gd_torq_in_py){
  vec3 gd_torq_in = numpyToArmaVector(gd_torq_in_py);
  plan_for_gendist = 1;
  gen_dist_torq = gd_torq_in;
}
void Satellite::remove_gendist_torq(){
  plan_for_gendist = 0;
  gen_dist_torq *= 0;
}



void Satellite::add_MTQ(vec3 body_ax, double max_moment, double cost){
  number_MTQ++;
  MTQ_axes.push_back(body_ax);
  MTQ_max.push_back(max_moment);
  MTQ_cost.push_back(cost);
  mtq_ax_mat = join_rows(mtq_ax_mat,body_ax);
}
void Satellite::add_MTQ_py(py::array_t<double> body_ax_py, double max_moment, double cost){
  vec3 body_ax = numpyToArmaVector(body_ax_py);
  number_MTQ++;
  MTQ_axes.push_back(body_ax);
  MTQ_max.push_back(max_moment);
  MTQ_cost.push_back(cost);
  mtq_ax_mat = join_rows(mtq_ax_mat,body_ax);
}
void Satellite::clear_MTQs(){
  number_MTQ = 0;
  MTQ_axes.clear();
  MTQ_max.clear();
  MTQ_cost.clear();
  mtq_ax_mat = mat(3,0).zeros();
}

void Satellite::add_magic(vec3 body_ax, double max_torq, double cost){
  number_magic++;
  magic_axes.push_back(body_ax);
  magic_max_torq.push_back(max_torq);
  magic_cost.push_back(cost);
  magic_ax_mat = join_rows(magic_ax_mat,body_ax);
}
void Satellite::add_magic_py(py::array_t<double> body_ax_py, double max_torq, double cost){
  vec3 body_ax = numpyToArmaVector(body_ax_py);
  number_magic++;
  magic_axes.push_back(body_ax);
  magic_max_torq.push_back(max_torq);
  magic_cost.push_back(cost);
  magic_ax_mat = join_rows(magic_ax_mat,body_ax);
}
void Satellite::clear_magics(){
  number_magic = 0;
  magic_axes.clear();
  magic_max_torq.clear();
  magic_cost.clear();
  magic_ax_mat = mat(3,0).zeros();
}

void Satellite::update_invJ_noRW(){
  // Compute J_noRW by subtracting RW contributions from Jcom
  // (removed unnecessary .t().t() operations which are identity)
  mat33 J_noRW = Jcom;
  for(int k=0;k<number_RW;k++)
  {
    J_noRW -= RW_axes[k]*RW_axes[k].t()*RW_J[k];
  }
  Jcom_noRW = J_noRW;
  invJcom_noRW = Jcom_noRW.i();
  invJcom = Jcom.i();
}

void Satellite::add_RW(vec3 body_ax, double J, double max_torq, double max_ang_mom, double cost,double AM_cost, double AM_cost_threshold,double stiction_cost, double stiction_threshold){
  number_RW++;
  RW_axes.push_back(body_ax);
  RW_max_torq.push_back(max_torq);
  RW_max_ang_mom.push_back(max_ang_mom);
  RW_cost.push_back(cost);
  RW_AM_cost.push_back(AM_cost);
  RW_AM_cost_threshold.push_back(AM_cost_threshold);
  RW_stiction_threshold.push_back(stiction_threshold);
  RW_stiction_cost.push_back(stiction_cost);
  RW_J.push_back(J);
  rw_ax_mat = join_rows(rw_ax_mat,body_ax);
  update_invJ_noRW();
}
void Satellite::add_RW_py(py::array_t<double> body_ax_py, double J, double max_torq, double max_ang_mom, double cost,double AM_cost, double AM_cost_threshold,double stiction_cost, double stiction_threshold){
  vec3 body_ax = numpyToArmaVector(body_ax_py);
  number_RW++;
  RW_axes.push_back(body_ax);
  RW_max_torq.push_back(max_torq);
  RW_max_ang_mom.push_back(max_ang_mom);
  RW_cost.push_back(cost);
  RW_AM_cost.push_back(AM_cost);
  RW_AM_cost_threshold.push_back(AM_cost_threshold);
  RW_stiction_threshold.push_back(stiction_threshold);
  RW_stiction_cost.push_back(stiction_cost);
  RW_J.push_back(J);
  rw_ax_mat = join_rows(rw_ax_mat,body_ax);
  update_invJ_noRW();
}
void Satellite::clear_RWs(){
  number_RW = 0;
  RW_axes.clear();
  RW_max_torq.clear();
  RW_max_ang_mom.clear();
  RW_cost.clear();
  RW_AM_cost.clear();
  RW_AM_cost_threshold.clear();
  RW_stiction_cost.clear();
  RW_stiction_threshold.clear();
  RW_J.clear();
  rw_ax_mat = mat(3,0).zeros();
}



void Satellite::set_AV_constraint(double wmax){
  useAVconstraint = true;
  AVmax = wmax;
}
void Satellite::clear_AV_constraint(){
  useAVconstraint = false;
  AVmax = nan("1");
}

void Satellite::add_sunpoint_constraint(vec3 body_ax, double angle,bool useACOS){
  number_sunpoints++;
  sunpoint_axes.push_back(body_ax);
  sunpoint_angs.push_back(angle);
  sunpoint_useACOSs.push_back(useACOS);
}
void Satellite::add_sunpoint_constraint_py(py::array_t<double> body_ax_py, double angle,bool useACOS){
  vec3 body_ax = numpyToArmaVector(body_ax_py);
  number_sunpoints++;
  sunpoint_axes.push_back(body_ax);
  sunpoint_angs.push_back(angle);
  sunpoint_useACOSs.push_back(useACOS);
}
void Satellite::clear_sunpoint_constraints(){
  number_sunpoints = 0;
  sunpoint_axes.clear();
  sunpoint_angs.clear();
  sunpoint_useACOSs.clear();
}

void Satellite::auto_scale_control_costs(double base_weight){
  // Scale control costs inversely proportional to max^2 (since cost is quadratic)
  // Preserves relative priority from existing input weights:
  //   new_cost = base_weight * input_weight / max^2
  // base_weight is an optional multiplier (default 1.0)

  // Validate inputs
  if(base_weight <= 0){
    std::cerr << "WARNING: auto_scale_control_costs called with non-positive base_weight="
              << base_weight << ", using 1.0" << std::endl;
    base_weight = 1.0;
  }

  if(number_MTQ == 0 && number_RW == 0 && number_magic == 0){
    std::cerr << "WARNING: auto_scale_control_costs called with no actuators defined" << std::endl;
    return;
  }

  // Scale MTQ costs: preserve input weight, divide by max^2
  for(int i = 0; i < number_MTQ; i++){
    double max_val = MTQ_max.at(i);
    if(max_val > 0){
      MTQ_cost.at(i) = base_weight * MTQ_cost.at(i) / (max_val * max_val);
    } else {
      std::cerr << "WARNING: MTQ " << i << " has non-positive max value, skipping scaling" << std::endl;
    }
  }

  // Scale RW costs
  for(int i = 0; i < number_RW; i++){
    double max_val = RW_max_torq.at(i);
    if(max_val > 0){
      RW_cost.at(i) = base_weight * RW_cost.at(i) / (max_val * max_val);
    } else {
      std::cerr << "WARNING: RW " << i << " has non-positive max torque, skipping scaling" << std::endl;
    }
  }

  // Scale magic actuator costs
  for(int i = 0; i < number_magic; i++){
    double max_val = magic_max_torq.at(i);
    if(max_val > 0){
      magic_cost.at(i) = base_weight * magic_cost.at(i) / (max_val * max_val);
    } else {
      std::cerr << "WARNING: Magic actuator " << i << " has non-positive max torque, skipping scaling" << std::endl;
    }
  }
}

vec Satellite::state_norm(vec x) const
{
  // Normalize quaternion portion of state vector
  // (removed unnecessary .t().t() copy - just copy directly)
  vec out = x;
  out(span(quat0index(),quat0index()+3)) = normalise(x(span(quat0index(),quat0index()+3)));
  return out;
}

/* This function gives the Jacobian of normalizing a system state
   Arguments:
    vec7 x is a 7x1 vector with the first 3 elements representing angular velocity and the last 4 representing a quaternion orientation, where the quaternion has not been normalised
  Returns:
    mat77 jac is a 7x7 matrix representing the jacobian of normalizing the quaternion in x
*/
mat Satellite::state_norm_jacobian(vec x) const
{
  vec4 q = x(span(quat0index(),quat0index()+3));
  double qn = norm(q);
  mat nj = mat(state_N(),state_N()).eye();
  double sq = 1.0;
  // sq *= sign(q(index_max(abs(q))));
  // if(q(0)!=0){
  //   sq *= sign(q(0));
  // }else if(q(1)!=0){
  //   sq *= sign(q(1));
  // }else if(q(2)!=0){
  //   sq *= sign(q(2));
  // }else if(q(3)!=0){
  //   sq *= sign(q(3));
  // }

  // Protect against division by zero/very small quaternion norms
  // This can happen with unstable trajectories during optimization
  const double qn_min = 1e-10;
  if(qn < qn_min) {
    // Return identity jacobian for degenerate case - the optimizer will reject this trajectory anyway
    return nj;
  }

  nj(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) = (mat44().eye()*(1.0/qn) - q*trans(q)/(qn*qn*qn))*sq;

  return nj;
}


cube Satellite::state_norm_hessian(vec x) const
{
  vec4 q = x(span(quat0index(),quat0index()+3));
  double qn = norm(q);
  cube out = cube(x.n_elem,x.n_elem,x.n_elem).zeros();

  // Protect against division by zero/very small quaternion norms
  const double qn_min = 1e-10;
  if(qn < qn_min) {
    return out;  // Return zero hessian for degenerate case
  }

  mat44 unitvecs4 = mat44().eye();
  for(int i = 0;i<4;i++){
    out.slice(quat0index()+i)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) = -(unitvecs4.col(i)*q.t() + q*unitvecs4.col(i).t() + q(i)*mat44().eye())/(qn*qn*qn) + 3.0*q(i)*q*q.t()/(qn*qn*qn*qn*qn);
  }

  // ei.t()*(mat44().eye()*(1.0/qn) - q*trans(q)/(qn*qn*qn))
  // ei.t()/qn - ei.t()*q*trans(q)/(qn*qn*qn)
  //
  // ei/qn - q*q.t()*ei/(qn*qn*qn)
  // ei*(-1/qn**2)*dqn__dq.t()
  //   - asscalar(q.t()*ei)*mat44().eye()/(qn*qn*qn)
  //   - q*ei.t()/(qn*qn*qn)
  //   - q*q.t()*ei*(-3/qn**4)*dqn__dq.t()
  //
  // sqrt(q.t()*q)
  // (1/qn)*q
  //
  // ei*(-1/qn**3)*q.t()
  //   - asscalar(q.t()*ei)*mat44().eye()/(qn*qn*qn)
  //   - q*ei.t()/(qn*qn*qn)
  //   - q*q.t()*ei*(-3/qn**5)*q.t()
  //
  //
  // -(1/qn**3)*(ei*q.t() + asscalar(q.t()*ei)*mat44().eye() + q*ei.t() - 3*asscalar(q.t()*ei)*q*q.t()/qn**2)




  // mat nj = mat(state_N(),state_N()).eye();
  // double sq = 1.0;
  // sq *= sign(q(index_max(abs(q))));
  // if(q(0)!=0){
  //   sq *= sign(q(0));
  // }else if(q(1)!=0){
  //   sq *= sign(q(1));
  // }else if(q(2)!=0){
  //   sq *= sign(q(2));
  // }else if(q(3)!=0){
  //   sq *= sign(q(3));
  // }

  return out;
}
/* This function returns a 6 x 7 matrix with a 3 x 3 identity matrix in the upper left,
and the transpose of the Wmat matrix for the quaternion in question in the bottom right. This
matrix can basically be used to transform a state vector from 7 x 1 to 6 x 1.
  Arguments:
    vec qk a 4x1 vector representing a unit quaternion
  Returns:
    mat gMat (6 x 7)
*/
mat Satellite::findGMat(vec4 qk) const
{
  //gmat = [eye(3) zeros(3,4);
          // zeros(3,3) wMat.'']
  mat out = mat(reduced_state_N(),state_N()).zeros();
  out(span(0,redang0index()-1),span(0,quat0index()-1)) = mat33().eye();
  out(redang0index(),quat0index(),size(3,4)) = findWMat(qk).t();
  if(state_N()>7){
    out(redang0index()+3,quat0index()+4,size(reduced_state_N()-6,state_N()-7)) = mat(state_N()-7,state_N()-7).eye();
  }
  // double q0 = qk(0);
  // double q1 = qk(1);
  // double q2 = qk(2);
  // double q3 = qk(3);
  //mat wMat = mat({{-q1, -q2, -q3}, {q0, -q3, q2}, {q3, q0, -q1}, {-q2, q1, q0}});
  //mat transWMat = mat({{-q1, q0, q3, -q2}, {-q2, -q3, q0, q1}, {-q3, q2, -q1, q0}});
  return out;//join_cols(join_rows(mat33().eye(), mat(3, 4).zeros()), join_rows(mat33().zeros(),findWMat(qk).t()));
}




/*This function returns constraints
  Arguments:
  int k should be time
  u should be u at a particular timestep
  umax should be maximum allowable u
 Returns:
  vec representing constraints
*/
vec Satellite::getConstraints(int k, int N, vec u, vec x, arma::vec3 sunECIvec) const
{
  //order: int(useAVconstraint) + number_sunpoints + 2*number_MTQ + 2*number_magic + (2+2+1)*number_RW
  vec ck = vec(constraint_N()).zeros();

  if(k <= N-1 && k >= 0)
  {
    int ind = 0;
    if(useAVconstraint){
      ck.row(ind) = (dot(x.rows(avindex0(),avindex0()+2),x.rows(avindex0(),avindex0()+2))-AVmax*AVmax)*(180.0/datum::pi)*(180.0/datum::pi)/pow(norm(AVmax),2.0);//put in degrees to make the magnitude bigger so more compatible with other costs/etc in alg and more readable.
      ind++;
    }

    vec4 qk = x.rows(quat0index(), quat0index()+3);
    for(int j=0;j<number_sunpoints;j++){
      if (!sunECIvec.is_zero()){
        if(sunpoint_useACOSs.at(j)){
          ck.row(ind) = sunpoint_angs.at(j)-acos(norm_dot(normalise(sunpoint_axes.at(j)),rotMat(qk).t()*normalise(sunECIvec)));
        } else {
          ck.row(ind) = (norm_dot((sunpoint_axes.at(j)),rotMat(qk).t()*(sunECIvec))-cos(sunpoint_angs.at(j)));
        }
      }
      ind++;
    }
    if(k<N-1){ //no control in last time step
      int ctrl_base = 0;
      for(int j=0;j<number_MTQ;j++){
        ck.row(ind) = (u.row(ctrl_base+j) - MTQ_max.at(j))/MTQ_max.at(j);
        ind++;
        ck.row(ind) = (-u.row(ctrl_base+j) - MTQ_max.at(j))/MTQ_max.at(j);
        ind++;
      }

      ctrl_base = number_MTQ;
      int state_base = 7;
      for(int j=0;j<number_RW;j++){
        ck.row(ind) = (u.row(ctrl_base+j)*MAGRW_TORQ_MULT - RW_max_torq.at(j))/RW_max_torq.at(j);
        ind++;
        ck.row(ind) = (-u.row(ctrl_base+j)*MAGRW_TORQ_MULT - RW_max_torq.at(j))/RW_max_torq.at(j);
        ind++;

        ck.row(ind) = (x.row(state_base+j) - RW_max_ang_mom.at(j))/RW_max_ang_mom.at(j);
        ind++;
        ck.row(ind) = (-x.row(state_base+j) - RW_max_ang_mom.at(j))/RW_max_ang_mom.at(j);
        ind++;

        ck.row(ind) = -pow(MAGRW_TORQ_MULT*u.row(ctrl_base+j)*x.row(state_base+j),2.0); //momentum and torq cannot both be zero! stiction. can pass through zero though.
        ind++;
      }

      ctrl_base = number_MTQ+number_RW;
      for(int j=0;j<number_magic;j++){
        ck.row(ind) = (u.row(ctrl_base+j)*MAGRW_TORQ_MULT - magic_max_torq.at(j))/magic_max_torq.at(j);
        ind++;
        ck.row(ind) = (-u.row(ctrl_base+j)*MAGRW_TORQ_MULT - magic_max_torq.at(j))/magic_max_torq.at(j);
        ind++;
      }

    }
    return ck;
  }
  else
  {
    throw new std::runtime_error("time bound error in inEqConstraints");
  }
  /*vec ineqC = ineqConstraints(t, tmax, u, umax);
  int eqC = eqConstraints();
  if(eqC==0)
  {
    return ineqC;
  } else {
    return join_cols(ineqC, vec(eqC));
  }*/
}
/* This function returns Jacobians for all constraints
   Arguments:
     int k should be time
     x is state at timestep k
     N = length of slew
  Returns:
    tuple<mat, mat> the Jacobian wrt u and the Jacobian wrt x
*/
tuple<mat,mat> Satellite::constraintJacobians(int k, int N, vec uk,vec xk,vec3 sunk) const
{
  //order: int(useAVconstraint) + number_sunpoints + 2*number_MTQ + 2*number_magic + (2+2+1)*number_RW
  vec ck = vec(constraint_N()).zeros();
  mat cku = mat(constraint_N(),control_N()).zeros();
  mat ckx = mat(constraint_N(),reduced_state_N()).zeros();

  int ind = 0;
  if(useAVconstraint){
    // c.row(ind) = (dot(x.rows(avindex0(),avindex0()+2),x.rows(avindex0(),avindex0()+2))-pow(AVmax,2))*(180.0/datum::pi)*(180.0/datum::pi);//put in degrees to make the magnitude bigger so more compatible with other costs/etc in alg.
    ckx(span(ind,ind),span(avindex0(),avindex0()+2)) = 2.0*trans(xk.rows(avindex0(),avindex0()+2))*(180.0/datum::pi)*(180.0/datum::pi)/(pow(norm(AVmax),2.0));
    ind++;
  }
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  for(int j=0;j<number_sunpoints;j++){
    if (!sunk.is_zero()){
      if(sunpoint_useACOSs.at(j)){
        // ck.row(ind) = sunpoint_angs.at(j)-acos(norm_dot(normalise(sunpoint_axes.at(j)),rotMat(qk).t()*normalise(sunECIvec)));
        double zz = dot(normalise(sunpoint_axes.at(j)),rotMat(qk).t()*normalise(sunk));
        zz = sign(zz)*min(abs(zz),sqrt(1.0 - (double) 1e-16));
        ckx(span(ind,ind),span(redang0index(),redang0index()+2)) = trans(normalise(sunpoint_axes.at(j)))*dRTBdqQ(qk, normalise(sunk))/sqrt(1-zz*zz);
      } else {
        // ck.row(ind) = (norm_dot((sunpoint_axes.at(j)),rotMat(qk).t()*(sunECIvec))-cos(sunpoint_angs.at(j)));
        ckx(span(ind,ind),span(redang0index(),redang0index()+2)) = trans(normalise(sunpoint_axes.at(j)))*dRTBdqQ(qk, normalise(sunk));
      }
    }
    ind++;
  }
  if(k<N-1){ //no control in last time step
    int ctrl_base = 0;
    if(number_MTQ>0){
      for(int j=0;j<number_MTQ;j++){
        // ck.row(ind) = u.row(ctrl_base+j) - MTQ_max.at(j);
        cku(ind,ctrl_base+j) = 1.0/MTQ_max.at(j);
        ind++;
        // ck.row(ind) = -u.row(ctrl_base+j) - MTQ_max.at(j);
        cku(ind,ctrl_base+j) = -1.0/MTQ_max.at(j);
        ind++;
      }
    }

    ctrl_base = number_MTQ;
    if(number_RW>0){
      int state_base = 7;
      for(int j=0;j<number_RW;j++){
        // ck.row(ind) = u.row(ctrl_base+j) - RW_max_torq.at(j);
        cku(ind,ctrl_base+j) = 1.0*MAGRW_TORQ_MULT/RW_max_torq.at(j);
        ind++;
        // ck.row(ind) = -u.row(ctrl_base+j) - RW_max_torq.at(j);
        cku(ind,ctrl_base+j) = -1.0*MAGRW_TORQ_MULT/RW_max_torq.at(j);
        ind++;

        // ck.row(ind) = x.row(state_base+j) - RW_max_ang_mom.at(j);
        ckx(ind,state_base-1+j) = 1.0/RW_max_ang_mom.at(j);
        ind++;
        // ck.row(ind) = -x.row(state_base+j) - RW_max_ang_mom.at(j);
        ckx(ind,state_base-1+j) = -1.0/RW_max_ang_mom.at(j);
        ind++;

        // ck.row(ind) = -pow(u.row(ctrl_base+j)*x.row(state_base+j),2); //momentum and torq cannot both be zero! stiction. can pass through zero though.
        ckx(ind,state_base-1+j) = -2.0*MAGRW_TORQ_MULT*MAGRW_TORQ_MULT*uk(ctrl_base+j)*xk(state_base+j)*uk(ctrl_base+j);
        cku(ind,ctrl_base+j) = -2.0*MAGRW_TORQ_MULT*MAGRW_TORQ_MULT*uk(ctrl_base+j)*xk(state_base+j)*xk(state_base+j);
        ind++;
      }
    }

    ctrl_base = number_MTQ+number_RW;
    if(number_magic>0){
      for(int j=0;j<number_magic;j++){
        // ck.row(ind) = u.row(ctrl_base+j) - magic_max_torq.at(j);
        cku(ind,ctrl_base+j) = 1.0*MAGRW_TORQ_MULT/magic_max_torq.at(j);
        ind++;
        // ck.row(ind) = -u.row(ctrl_base+j) - magic_max_torq.at(j);
        cku(ind,ctrl_base+j) = -1.0*MAGRW_TORQ_MULT/magic_max_torq.at(j);
        ind++;
      }
    }
  }
  return std::make_tuple(cku, ckx);
}
/* This function returns Hessians for all constraints
   Arguments:
     int k should be time
     x is state at timestep k
     N = length of slew
  Returns:
    tuple<cube,cube,cube> the Hessians wrt uu,ux,xx
*/
tuple<cube,cube,cube> Satellite::constraintHessians(int k, int N, vec uk,vec xk,vec3 sunk) const
{

  cube ckuu = zeros<cube>(control_N(),control_N(),constraint_N());
  cube ckux = zeros<cube>(control_N(),reduced_state_N(),constraint_N());
  cube ckxx = zeros<cube>(reduced_state_N(),reduced_state_N(),constraint_N());


  int ind = 0;
  if(useAVconstraint){
    // c.row(ind) = (dot(x.rows(avindex0(),avindex0()+2),x.rows(avindex0(),avindex0()+2))-pow(AVmax,2))*(180.0/datum::pi)*(180.0/datum::pi);//put in degrees to make the magnitude bigger so more compatible with other costs/etc in alg.
    ckxx(span(avindex0(),avindex0()+2),span(avindex0(),avindex0()+2),span(ind,ind)) = 2*mat33().eye()*(180.0/datum::pi)*(180.0/datum::pi)/(pow(norm(AVmax),2.0));
    ind++;
  }
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  for(int j=0;j<number_sunpoints;j++){
    if (!sunk.is_zero()){
      if(sunpoint_useACOSs.at(j)){
        // ck.row(ind) = sunpoint_angs.at(j)-acos(norm_dot(normalise(sunpoint_axes.at(j)),rotMat(qk).t()*normalise(sunECIvec)));
        double zz = dot(normalise(sunpoint_axes.at(j)),rotMat(qk).t()*normalise(sunk));
        zz = sign(zz)*min(abs(zz),sqrt(1.0 - (double) 1e-16));
        vec dadq = (trans(normalise(sunpoint_axes.at(j)))*dRTBdqQ(qk, normalise(sunk))/sqrt(1.0-zz*zz)).t();
        ckxx(redang0index(),redang0index(),ind,arma::size(3,3,1))  = ddvTRTudqQ(qk,normalise(sunpoint_axes.at(j)), normalise(sunk))/sqrt(1.0-zz*zz) + dadq*(1.0/sqrt(1.0-zz*zz))*zz*dadq.t();

      } else {
        // ck.row(ind) = (norm_dot((sunpoint_axes.at(j)),rotMat(qk).t()*(sunECIvec))-cos(sunpoint_angs.at(j)));
        ckxx(redang0index(),redang0index(),ind,size(3,3,1)) = ddvTRTudqQ(qk,normalise(sunpoint_axes.at(j)),normalise(sunk));
      }
    }
    ind++;
  }
  if(k<N-1){ //no control in last time step
    int ctrl_base = 0;
    if(number_MTQ>0){
      for(int j=0;j<number_MTQ;j++){
        // ck.row(ind) = u.row(ctrl_base+j) - MTQ_max.at(j);
        ind++;
        // ck.row(ind) = -u.row(ctrl_base+j) - MTQ_max.at(j);
        ind++;
      }
    }

    ctrl_base = number_MTQ;
    int state_base = 7;
    if(number_RW>0){
      for(int j=0;j<number_RW;j++){
        // ck.row(ind) = u.row(ctrl_base+j) - RW_max_torq.at(j);
        ind++;
        // ck.row(ind) = -u.row(ctrl_base+j) - RW_max_torq.at(j);
        ind++;

        // ck.row(ind) = x.row(state_base+j) - RW_max_ang_mom.at(j);
        ind++;
        // ck.row(ind) = -x.row(state_base+j) - RW_max_ang_mom.at(j);
        ind++;

        // ck.row(ind) = -pow(u.row(ctrl_base+j)*x.row(state_base+j),2); //momentum and torq cannot both be zero! stiction. can pass through zero though.
        ckxx(state_base+j-1,state_base+j-1,ind) = MAGRW_TORQ_MULT*MAGRW_TORQ_MULT*-2.0*uk(ctrl_base+j)*uk(ctrl_base+j);
        ckuu(ctrl_base+j,ctrl_base+j,ind) = MAGRW_TORQ_MULT*MAGRW_TORQ_MULT*-2.0*xk(state_base+j)*xk(state_base+j);
        ckux(ctrl_base+j,state_base+j-1,ind) = MAGRW_TORQ_MULT*MAGRW_TORQ_MULT*-4.0*xk(state_base+j)*uk(ctrl_base+j);
        ind++;
      }
    }
  }

  return std::make_tuple(ckuu*constraint_hess_mult, ckux*constraint_hess_mult,ckxx*constraint_hess_mult);
}




double Satellite::stepcost_vec(int k, int N, vec xk, vec uk,vec ukprev, vec3 satvec_k, vec3 ECIvec_k,vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const
{
  COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
  double w_ang = get<0>(costSettings_tmp);
  double w_av = get<1>(costSettings_tmp);
  double w_u_mult = get<2>(costSettings_tmp);
  double w_avmag = get<3>(costSettings_tmp);
  double w_avang = get<4>(costSettings_tmp);

  int whichAngCostFunc = get<9>(costSettings_tmp);
  int useRawControlCost = get<10>(costSettings_tmp);

  mat act_cost_mat = mat(control_N(),control_N()).zeros();

  vec3 magvec = vec(3).zeros();
  if(k<N-1){
    if(number_MTQ>0)
    {
      act_cost_mat(0,0,size(number_MTQ,number_MTQ)) = diagmat(vec(MTQ_cost));
      magvec = mtq_ax_mat*uk.head(number_MTQ);
    }
    if(number_RW>0)
    {
      act_cost_mat(number_MTQ,number_MTQ,size(number_RW,number_RW)) = diagmat(vec(RW_cost));
    }
    if(number_magic>0)
    {
      act_cost_mat(number_MTQ+number_RW,number_MTQ+number_RW,size(number_magic,number_magic)) = diagmat(vec(magic_cost));
    }
  }else{
    w_u_mult = 0;
    w_ang = get<5>(costSettings_tmp);
    w_av = get<6>(costSettings_tmp);
    w_avmag = get<7>(costSettings_tmp);
    w_avang = get<8>(costSettings_tmp);
  }
  xk = state_norm(xk);
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  vec3 wk = xk.rows(avindex0(),avindex0()+2);

  double angcost = 0.0;

  // Cache rotation matrix - called multiple times with same quaternion
  mat33 Rk = rotMat(qk);
  mat33 RkT = Rk.t();

  //if k == N, lku is zero so lkuu and lkux also zeros
  vec3 sk = normalise(satvec_k);
  vec3 ek = normalise(ECIvec_k);

  vec3 RkT_ek = RkT*ek;  // Reuse this product
  double ddot = norm_dot(sk, RkT_ek);
  ddot = min(max(ddot, -1.0), 1.0);
  vec3 angerrvec = cross(RkT_ek, sk);
  switch(whichAngCostFunc){
      case 0:
        angcost = 1.0-ddot;
        break;
      case 1:
        angcost = 0.5*(1.0-ddot)*(1.0-ddot);
        break;
      case 2:
        angcost = acos(ddot);
        break;
      case 3:
        angcost = 0.5*pow(acos(ddot),2.0);
        break;
      default:
        throw("incorrect ang cost function specifier");
    }
  double cross_cost = w_avang*dot(wk,angerrvec);


  angcost *= w_ang;
  vec3 nb = RkT*normalise(BECI_k);  // Reuse cached RkT
  double state_cost = 0.5*as_scalar(wk.t()*wk)*w_av + angcost;//*(1.0-ddot));//0.5*as_scalar(wk.t()*wk*w_av + w_ang*phi*phi);//0.5*as_scalar(wk.t()*wk*w_av + w_ang*(1-ddot)*(1-ddot));//0.5*as_scalar(vsk.t()*Wxx*vsk);
  double actuation_cost = 0.0;
  double ang_mom_cost = 0.0;
  double stiction_cost = 0.0;

  if (useRawControlCost==0){
    actuation_cost = 0.5*as_scalar((uk-ukprev).t()*act_cost_mat*(uk-ukprev))*w_u_mult;
  }else{
    actuation_cost = 0.5*as_scalar(uk.t()*act_cost_mat*uk)*w_u_mult;
  }
  double state_mag_cost = w_avmag*abs(dot(wk,nb));
  double act_mag_cost = 0.0;

  if(number_RW>0){
    for(int j = 0;j<number_RW;j++)
    {
      double z = xk(7+j);
      double sz = sign(z);
      ang_mom_cost += 0.5*RW_AM_cost.at(j)*pow(shifted_softplus(z*sz,RW_AM_cost_threshold.at(j)),2);
      stiction_cost += 0.5*RW_stiction_cost.at(j)*pow(smoothstep((RW_stiction_threshold.at(j)-z*sz)/RW_stiction_threshold.at(j))*RW_stiction_threshold.at(j),2);
    }

  }
  return state_cost + cross_cost + actuation_cost + state_mag_cost + act_mag_cost + ang_mom_cost + stiction_cost;
}


double Satellite::stepcost_quat(int k, int N, vec xk, vec uk,vec ukprev, vec3 satvec_k, vec4 ECIvec_k,vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const
{
  COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
  double w_ang = get<0>(costSettings_tmp);
  double w_av = get<1>(costSettings_tmp);
  double w_u_mult = get<2>(costSettings_tmp);
  double w_avmag = get<3>(costSettings_tmp);
  double w_avang = get<4>(costSettings_tmp);

  int whichAngCostFunc = get<9>(costSettings_tmp);
  int useRawControlCost = get<10>(costSettings_tmp);

  mat act_cost_mat = mat(control_N(),control_N()).zeros();

  vec3 magvec = vec(3).zeros();
  if(k<N-1){
    if(number_MTQ>0)
    {
      act_cost_mat(0,0,size(number_MTQ,number_MTQ)) = diagmat(vec(MTQ_cost));
      magvec = mtq_ax_mat*uk.head(number_MTQ);
    }
    if(number_RW>0)
    {
      act_cost_mat(number_MTQ,number_MTQ,size(number_RW,number_RW)) = diagmat(vec(RW_cost));
    }
    if(number_magic>0)
    {
      act_cost_mat(number_MTQ+number_RW,number_MTQ+number_RW,size(number_magic,number_magic)) = diagmat(vec(magic_cost));
    }
  }else{
    w_u_mult = 0;
    w_ang = get<5>(costSettings_tmp);
    w_av = get<6>(costSettings_tmp);
    w_avmag = get<7>(costSettings_tmp);
    w_avang = get<8>(costSettings_tmp);
  }
  xk = state_norm(xk);
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  vec3 wk = xk.rows(avindex0(),avindex0()+2);

  double angcost = 0.0;

  // Cache rotation matrix transpose - used for magnetic field direction
  mat33 RkT = rotMat(qk).t();

  //if k == N, lku is zero so lkuu and lkux also zeros
  vec3 sk = normalise(satvec_k);
  vec4 ek = normalise(ECIvec_k);

  vec4 qerr = normquaterr(ek,qk);
  vec3 angerrvec = qerr(span(1,3));
  double ddot = norm_dot(ek,qk);

  mat::fixed<4,3> Wq = findWMat(qk);
  mat::fixed<4,3> We = findWMat(ek);

  // Cache We*We.t() product for cost functions 1 and 3
  mat::fixed<4,4> WeWeT = We*We.t();

  switch(whichAngCostFunc){
    case 0:
      angcost = 1.0-abs(ddot);
      break;
    case 1:
      angcost = 0.5*as_scalar(qk.t()*WeWeT*qk);
      break;
    case 2:
      // Cayley-like cost: 0.5 * ||qerr_vec||^2 / qerr_scalar^2
      // Note: w_ang is applied after the switch, so don't include it here
      angcost = 0.5*dot(angerrvec,angerrvec);
      if(abs(qerr(0))>EPSVAR){
          angcost *= 1.0/(qerr(0)*qerr(0));
        }else if(qerr(0)==0){
          angcost *= 1.0/EPSVAR;
        } else {
          angcost *= 1.0/(EPSVAR*sign(qerr(0)));
        }
      break;
    case 3:
      angcost = 0.5*as_scalar(qk.t()*WeWeT*qk);
      break;
    case 4:
      angcost = 1.0-(ddot*ddot);
      break;
    default:
      throw("incorrect ang cost function specifier.0-4 allowed for 3d orientation specification.");
    }
  double cross_cost = -sign(ddot)*as_scalar(ek.t()*Wq*wk)*w_avang;

  angcost *= w_ang;
  vec3 nb = RkT*normalise(BECI_k);  // Reuse cached RkT
  double state_cost = 0.5*as_scalar(wk.t()*wk)*w_av + angcost;//*(1.0-ddot));//0.5*as_scalar(wk.t()*wk*w_av + w_ang*phi*phi);//0.5*as_scalar(wk.t()*wk*w_av + w_ang*(1-ddot)*(1-ddot));//0.5*as_scalar(vsk.t()*Wxx*vsk);
  double actuation_cost = 0.0;
  double ang_mom_cost = 0.0;
  double stiction_cost = 0.0;

  if (useRawControlCost==0){
    actuation_cost = 0.5*as_scalar((uk-ukprev).t()*act_cost_mat*(uk-ukprev))*w_u_mult;
  }else{
    actuation_cost = 0.5*as_scalar(uk.t()*act_cost_mat*uk)*w_u_mult;
  }
  double state_mag_cost = w_avmag*abs(dot(wk,nb));
  double act_mag_cost = 0.0;
  if(number_RW>0){
    for(int j = 0;j<number_RW;j++)
    {
      double z = xk(7+j);
      double sz = sign(z);
      ang_mom_cost += 0.5*RW_AM_cost.at(j)*pow(shifted_softplus(z*sz,RW_AM_cost_threshold.at(j)),2);
      stiction_cost += 0.5*RW_stiction_cost.at(j)*pow(smoothstep((RW_stiction_threshold.at(j)-z*sz)/RW_stiction_threshold.at(j))*RW_stiction_threshold.at(j),2);
    }
  }
  return state_cost + cross_cost + actuation_cost + state_mag_cost + act_mag_cost + ang_mom_cost + stiction_cost;
}

// ============================================================================
// Helper functions for cost Jacobian computation
// ============================================================================

ExtractedCostSettings ExtractedCostSettings::fromTuple(const COST_SETTINGS_FORM& settings) {
    ExtractedCostSettings s;
    s.w_ang = std::get<0>(settings);
    s.w_av = std::get<1>(settings);
    s.w_u_mult = std::get<2>(settings);
    s.w_avmag = std::get<3>(settings);
    s.w_avang = std::get<4>(settings);
    s.w_ang_N = std::get<5>(settings);
    s.w_av_N = std::get<6>(settings);
    s.w_avmag_N = std::get<7>(settings);
    s.w_avang_N = std::get<8>(settings);
    s.whichAngCostFunc = std::get<9>(settings);
    s.useRawControlCost = std::get<10>(settings);
    s.useFullCostHess = std::get<11>(settings);
    return s;
}

void ExtractedCostSettings::applyTerminalWeights() {
    w_u_mult = 0.0;
    w_ang = w_ang_N;
    w_av = w_av_N;
    w_avmag = w_avmag_N;
    w_avang = w_avang_N;
}

mat Satellite::setupActuationCostMatrix(int k, int N, ExtractedCostSettings& settings) const {
    mat act_cost_mat = mat(control_N(), control_N()).zeros();

    if (k < N-1) {
        if (number_MTQ > 0) {
            act_cost_mat(0, 0, size(number_MTQ, number_MTQ)) = diagmat(vec(MTQ_cost));
        }
        if (number_RW > 0) {
            act_cost_mat(number_MTQ, number_MTQ, size(number_RW, number_RW)) = diagmat(vec(RW_cost));
        }
        if (number_magic > 0) {
            act_cost_mat(number_MTQ + number_RW, number_MTQ + number_RW, size(number_magic, number_magic)) = diagmat(vec(magic_cost));
        }
    } else {
        settings.applyTerminalWeights();
    }

    return act_cost_mat;
}

RWCostResults Satellite::computeRWCostsAndJacobians(
    const vec& xk,
    vec& lkx,
    mat& lkxx
) const {
    RWCostResults result = {0.0, 0.0};

    if (number_RW <= 0) return result;

    for (int j = 0; j < number_RW; j++) {
        // CRITICAL: Read from FULL state at index 7+j
        double z = xk(7 + j);
        double sz = sign(z);

        // Angular momentum cost (shifted softplus)
        double sp = shifted_softplus(z * sz, RW_AM_cost_threshold.at(j));
        double spd = shifted_softplus_deriv(z * sz, RW_AM_cost_threshold.at(j));
        double spdd = shifted_softplus_deriv2(z * sz, RW_AM_cost_threshold.at(j));

        // CRITICAL: Write to REDUCED gradient at index 6+j
        if (isinf(sp)) {
            result.ang_mom_cost += 0.5 * RW_AM_cost.at(j) * sp * sp;
            lkx(6 + j) += sz * RW_AM_cost.at(j) * 1 * sp;
            lkxx(6 + j, 6 + j) += RW_AM_cost.at(j) * 1;
        } else {
            result.ang_mom_cost += 0.5 * RW_AM_cost.at(j) * pow(sp, 2);
            lkx(6 + j) += sz * RW_AM_cost.at(j) * sp * spd;
            lkxx(6 + j, 6 + j) += RW_AM_cost.at(j) * (spd * spd + sp * spdd);
        }

        // Stiction cost (smoothstep)
        double threshold = RW_stiction_threshold.at(j);
        double stic_arg = (threshold - z * sz) / threshold;
        double ss = smoothstep(stic_arg);
        double ssd = smoothstep_deriv(stic_arg);

        result.stiction_cost += 0.5 * RW_stiction_cost.at(j) * pow(ss * threshold, 2.0);

        // CRITICAL: Write to REDUCED gradient at index 6+j
        lkx(6 + j) += RW_stiction_cost.at(j) * (ss * threshold * ssd * threshold * (-sz / threshold));

        lkxx(6 + j, 6 + j) += RW_stiction_cost.at(j) * pow(ssd, 2.0);
    }

    return result;
}

// ============================================================================
// Cost Jacobian Functions
// ============================================================================

cost_jacs  Satellite::veccostJacobians(int k, int N, vec xk, vec uk,vec ukprev, vec3 satvec_k, vec3 ECIvec_k,vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const
{
  ExtractedCostSettings settings = ExtractedCostSettings::fromTuple(*costSettings_ptr);

  // Setup actuation cost matrix (handles terminal step weight update)
  mat act_cost_mat = setupActuationCostMatrix(k, N, settings);
  vec3 magvec = vec(3).zeros();
  if (k < N-1 && number_MTQ > 0) {
    magvec = mtq_ax_mat*uk.head(number_MTQ);
  }

  // Extract local copies of weights (may have been updated by setupActuationCostMatrix for terminal step)
  double w_ang = settings.w_ang;
  double w_av = settings.w_av;
  double w_u_mult = settings.w_u_mult;
  double w_avmag = settings.w_avmag;
  double w_avang = settings.w_avang;
  int whichAngCostFunc = settings.whichAngCostFunc;
  int useRawControlCost = settings.useRawControlCost;

  if(ECIvec_k.is_zero())
  {
    w_ang = 0.0;
  }



  vec lkx = vec(reduced_state_N()).zeros();
  mat lkxx = mat(reduced_state_N(),reduced_state_N()).zeros();
  mat lkux = mat(control_N(),reduced_state_N()).zeros();
  vec lku = vec(control_N()).zeros();
  mat lkuu = mat(control_N(),control_N()).zeros();

  vec xkraw = xk;
  xk = state_norm(xk);
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  vec4 qkraw = xkraw.rows(quat0index(), quat0index()+3);
  mat nj = findGMat(qk)*state_norm_jacobian(xkraw)*findGMat(qkraw).t();


  // vec4 qk = normalise(xk.rows(quat0index(), quat0index()+3));
  vec3 wk = xk(span(avindex0(),avindex0()+2));
  //vec v = satAlignVector;//get<1>(vu);
  vec3 sk = normalise(satvec_k);
  vec3 ek = normalise(ECIvec_k);

  mat::fixed<4,3> Wq = findWMat(qk);
  double ddot = norm_dot(sk,rotMat(qk).t()*ek);
  vec3 angerrvec = (cross(rotMat(qk).t()*ek,sk));

  double sc_ang;
  vec3 d_sc_ang;
  mat33 dd_sc_ang;

  tuple<double, vec3,mat33> angres = cost2angQ(qk,sk,ek);
  double phi = get<0>(angres);
  vec3 dphi = get<1>(angres);
  mat33 ddphi = get<2>(angres);

  // vec case3_eigs;
  switch (whichAngCostFunc) {
      case 0:
        sc_ang = (1.0-ddot);
        d_sc_ang = -1.0*(sk.t()*dRTBdqQ(qk,ek)).t();
        dd_sc_ang = -1.0*(ddvTRTudqQ(qk,sk,ek));
        break;
      case 1:
        sc_ang = 0.5*(1.0-ddot)*(1.0-ddot);
        d_sc_ang = (1.0-ddot)*-1.0*(sk.t()*dRTBdqQ(qk,ek)).t();
        dd_sc_ang = ((1.0-ddot)*-1.0*ddvTRTudqQ(qk,sk,ek) + (sk.t()*dRTBdqQ(qk,ek)).t()*(sk.t()*dRTBdqQ(qk,ek)));
        break;
      case 2:
        // sc_ang = w_ang*acos(ddot);
        sc_ang = phi;
        d_sc_ang = dphi;
        // {
        //   vec eigs;
        //   mat vecs;
        //   ddphi = 0.5*(ddphi + ddphi.t()); // symmetrize
        //   eig_sym(eigs, vecs, ddphi);
        //   eigs = clamp(eigs, 0.0, datum::inf);
        //   dd_sc_ang = vecs * diagmat(eigs) * vecs.t();
        // }
        dd_sc_ang = ddphi;
        break;
      case 3:
        // angcost = 0.5*w_ang*pow(acos(ddot),2.0);
        sc_ang =  0.5*phi*phi;
        d_sc_ang = dphi*phi;
        dd_sc_ang = dphi*dphi.t();//(dphi*dphi.t() + ddphi*phi);
        // vec3 dphi = get<1>(angres);
        // mat33 ddphi = get<2>(angres);
        // eig_sym(case3_eigs, dphi*dphi.t());
        break;
      default:
        throw("incorrect ang cost function specifier");
    }
    switch (whichAngCostFunc) {
    // ... cases ...
}

// // ADD THIS RIGHT HERE:

  vec3 nb = rotMat(qk).t()*normalise(BECI_k);
  mat::fixed<3,3> dBdq = dRTBdqQ(qk,normalise(BECI_k));
  mat::fixed<3,3> ddBwdq = ddvTRTudqQ(qk,wk,normalise(BECI_k));
  // mat::fixed<3,3> ddBmdq = ddvTRTudqQ(qk,magvec,normalise(BECI_k));
  cube::fixed<3,3,3> ddBdq = ddRTudqQ(qk,normalise(BECI_k));
  double state_cost = 0.5*as_scalar(wk.t()*wk)*w_av + w_ang*sc_ang;
  lkxx(0,0,size(3,3)) += mat33().eye()*w_av;
  lkx.head(3) += wk*w_av;
  lkxx(redang0index(),redang0index(),size(3,3)) += w_ang*dd_sc_ang;
  lkx(span(redang0index(),redang0index()+2)) += w_ang*d_sc_ang;

  double cross_cost = w_avang*dot(wk,angerrvec);
  lkx.head(3) += w_avang*angerrvec;
  lkx(span(redang0index(),redang0index()+2)) += -w_avang*(skewSymmetric(sk)*dRTBdqQ(qk,ek)).t()*wk;
  lkxx(0,redang0index(),size(3,3)) += -w_avang*skewSymmetric(sk)*dRTBdqQ(qk,ek);
  lkxx(redang0index(),redang0index(),size(3,3)) += w_avang*ddvTRTudqQ(qk,cross(sk,wk),ek);
  lkxx(redang0index(),0,size(3,3)) += -w_avang*(skewSymmetric(sk)*dRTBdqQ(qk,ek)).t();


  

  double actuation_cost = 0.0;
  double ang_mom_cost = 0.0;
  double stiction_cost = 0.0;

  if(k>0){
    if (useRawControlCost==0){
      lku += act_cost_mat*(uk-ukprev)*w_u_mult;
    }else{
      lku += act_cost_mat*(uk)*w_u_mult;
    }
    lkuu += act_cost_mat*w_u_mult;
  }else{
    if (useRawControlCost!=0){
      lku += act_cost_mat*(uk)*w_u_mult;
      lkuu += act_cost_mat*w_u_mult;
    }
  }
  double state_mag_cost = w_avmag*abs(dot(wk,nb));
  double savang = sign(dot(wk,nb));
  lkx.head(3) += w_avmag*savang*nb;
  lkx(span(redang0index(),redang0index()+2)) += w_avmag*savang*(wk.t()*dBdq).t();
  lkxx(0,redang0index(),size(3,3)) += w_avmag*savang*dBdq;
  lkxx(redang0index(),redang0index(),size(3,3)) += w_avmag*savang*ddBwdq;
  lkxx(redang0index(),0,size(3,3)) += w_avmag*savang*dBdq.t();

  


  // mat33 cross_hess = w_avang*ddvTRTudqQ(qk,cross(sk,wk),ek);

  // mat33 avmag_hess = w_avmag*savang*ddBwdq;

  // vec eigs_ddphi;
  // eig_sym(eigs_ddphi, ddphi);

  // // Check eigenvalues of each
  // vec eigs_ang, eigs_cross, eigs_avmag;
  // eig_sym(eigs_ang, w_ang*dd_sc_ang);
  // eig_sym(eigs_cross, cross_hess);
  // eig_sym(eigs_avmag, avmag_hess);



  double act_mag_cost = 0.0;

  // RW costs using helper function
  RWCostResults rwCosts = computeRWCostsAndJacobians(xk, lkx, lkxx);
  ang_mom_cost += rwCosts.ang_mom_cost;
  stiction_cost += rwCosts.stiction_cost;

  cost_jacs out;
  out.lx = nj.t()*lkx;
  out.lxx = nj.t()*lkxx*nj;
  out.lux = lkux*nj;
  out.lu = lku;
  out.luu = lkuu;
  return out;
}


cost_jacs  Satellite::quatcostJacobians(int k, int N, vec xk, vec uk,vec ukprev, vec3 satvec_k, vec4 ECIvec_k,vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const
{
  ExtractedCostSettings settings = ExtractedCostSettings::fromTuple(*costSettings_ptr);

  // Setup actuation cost matrix (handles terminal step weight update)
  mat act_cost_mat = setupActuationCostMatrix(k, N, settings);
  vec3 magvec = vec(3).zeros();
  if (k < N-1 && number_MTQ > 0) {
    magvec = mtq_ax_mat*uk.head(number_MTQ);
  }

  // Extract local copies of weights (may have been updated by setupActuationCostMatrix for terminal step)
  double w_ang = settings.w_ang;
  double w_av = settings.w_av;
  double w_u_mult = settings.w_u_mult;
  double w_avmag = settings.w_avmag;
  double w_avang = settings.w_avang;
  int whichAngCostFunc = settings.whichAngCostFunc;
  int useRawControlCost = settings.useRawControlCost;

  if(ECIvec_k.is_zero())
  {
    w_ang = 0.0;
  }



  vec lkx = vec(reduced_state_N()).zeros();
  mat lkxx = mat(reduced_state_N(),reduced_state_N()).zeros();
  mat lkux = mat(control_N(),reduced_state_N()).zeros();
  vec lku = vec(control_N()).zeros();
  mat lkuu = mat(control_N(),control_N()).zeros();


  vec xkraw = xk;
  xk = state_norm(xk);
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  vec4 qkraw = xkraw.rows(quat0index(), quat0index()+3);
  mat nj = findGMat(qk)*state_norm_jacobian(xkraw)*findGMat(qkraw).t();
  // vec4 qk = (xk.rows(quat0index(), quat0index()+3));
  // vec4 qk = normalise(xk.rows(quat0index(), quat0index()+3));
  vec3 wk = xk.rows(avindex0(),avindex0()+2);
  //vec v = satAlignVector;//get<1>(vu);
  vec3 sk = normalise(satvec_k);
  vec4 ek = normalise(ECIvec_k);

  // double sq = sign(qk(0));
  // if(sq==0){sq=1;}
  // double se = sign(ek(0));
  // if(se==0){se=1;}
  // qk *= se*sq;

  mat::fixed<4,3> Wq = findWMat(qk);
  mat::fixed<4,3> We = findWMat(ek);

  vec4 qerr = normquaterr(ek,qk);//normalise(join_cols(vec(1).ones()*norm_dot(ek,qk),ek(0)*qk.rows(1,3) - qk(0)*ek.rows(1,3)-cross(ek.rows(1,3),qk.rows(1,3))));
  vec3 angerrvec = qerr(span(1,3));
  mat33 d_angerrvec = join_rows(-ek.rows(1,3), ek(0)*mat33().eye() - skewSymmetric(ek.rows(1,3)))*Wq;
  double ddot = norm_dot(qk,ek);
  double sc_ang;
  vec3 d_sc_ang;
  mat33 dd_sc_ang;

  switch(whichAngCostFunc){
    case 0:
      sc_ang = 1.0-abs(ddot);
      d_sc_ang = -sign(ddot)*(ek.t()*Wq).t();
      if(ddot==0){
        d_sc_ang = -(ek.t()*Wq).t();
      }
      dd_sc_ang = mat33().eye()*abs(ddot);//Wq.t()*Wq*abs(ddot);//abs(ddot)*mat33().eye();//mat33().eye();//

      break;
    case 1:
      sc_ang = 0.5*as_scalar(qk.t()*We*We.t()*qk);
      d_sc_ang = Wq.t()*We*We.t()*qk;//We.t()*qk;//*sign(ek(0))*sq;//Wq.t()*We*We.t()*qk;
      dd_sc_ang = Wq.t()*We*We.t()*Wq - mat33().eye()*dot(qk,We*We.t()*qk);;// - mat33().eye()*dot(qk,We*We.t()*qk);;//mat33().eye();//Wq.t()*We*We.t()*Wq;//mat33().eye();// Wq.t()*We*We.t()*Wq;// - mat33().eye()*dot(qk,We*We.t()*qk);
      break;
    case 2:
      // Cayley-like cost: 0.5 * ||qerr_vec||^2 / qerr_scalar^2
      // Note: w_ang is applied after the switch (state_cost = ... + w_ang*sc_ang)
      sc_ang = 0.5*dot(angerrvec,angerrvec);
      if(abs(qerr(0))>EPSVAR){
        sc_ang *= 1.0/(qerr(0)*qerr(0));
        d_sc_ang = angerrvec/qerr(0);
      }else if(qerr(0)==0){
        sc_ang *= 1.0/EPSVAR;
        d_sc_ang = angerrvec/(EPSVAR);
      } else {
        sc_ang *= 1.0/(EPSVAR*sign(qerr(0)));
        d_sc_ang = angerrvec/(EPSVAR*sign(qerr(0)));
      }
      dd_sc_ang = mat33().eye();

      break;
    case 3:
      sc_ang = 0.5*as_scalar(qk.t()*We*We.t()*qk);
      d_sc_ang = Wq.t()*We*We.t()*qk;//We.t()*qk;//*sign(ek(0))*sq;//Wq.t()*We*We.t()*qk;
      dd_sc_ang = Wq.t()*We*We.t()*Wq;//mat33().eye();//Wq.t()*We*We.t()*Wq;//mat33().eye();// Wq.t()*We*We.t()*Wq;// - mat33().eye()*dot(qk,We*We.t()*qk);
      break;
    case 4:
      sc_ang = 1.0-(ddot*ddot);
      d_sc_ang = -2.0*ddot*(ek.t()*Wq).t();
      dd_sc_ang = -2.0*(ek.t()*Wq).t()*ek.t()*Wq + 2.0*ddot*ddot*mat33().eye();
      break;
    case 5:
      sc_ang = 1.0-(ddot*ddot);
      d_sc_ang = -2.0*ddot*(ek.t()*Wq).t();
      dd_sc_ang = -2.0*(ek.t()*Wq).t()*ek.t()*Wq;
      break;
    default:
      throw("incorrect ang cost function specifier.0-5 allowed for 3d orientation specification.");
    }
  // sc_ang = 0.5*as_scalar((qk-ek).t()*Wq*Wq.t()*(qk-ek));
  // if(abs(qerr(0))>EPSVAR)
  // {
  //   sc_ang = 0.5*dot(angerrvec,angerrvec)/(qerr(0)*qerr(0));
  // }else{
  //   if(qerr(0)==0){sc_ang = 0.5*dot(angerrvec,angerrvec)/(EPSVAR);}else{sc_ang = 0.5*dot(angerrvec,angerrvec)/(EPSVAR*sign(qerr(0)));}
  // }

  // d_sc_ang = Wq.t()*(qk-ek);//We.t()*qk;//*sign(ek(0))*sq;//Wq.t()*We*We.t()*qk;
  //
  // dd_sc_ang = Wq.t()*Wq*Wq.t()*Wq;// - mat33().eye()*dot(qk,Wq*Wq.t()*(qk-ek));;//mat33().eye();//Wq.t()*We*We.t()*Wq;//mat33().eye();// Wq.t()*We*We.t()*Wq;// - mat33().eye()*dot(qk,We*We.t()*qk);
  // dd_sc_ang = mat33().eye();

  vec3 nb = rotMat(qk).t()*normalise(BECI_k);
  mat::fixed<3,3> dBdq = dRTBdqQ(qk,normalise(BECI_k));
  mat::fixed<3,3> ddBwdq = ddvTRTudqQ(qk,wk,normalise(BECI_k));
  // mat::fixed<3,3> ddBmdq = ddvTRTudqQ(qk,magvec,normalise(BECI_k));
  cube::fixed<3,3,3> ddBdq = ddRTudqQ(qk,normalise(BECI_k));

  double state_cost = 0.5*as_scalar(wk.t()*wk)*w_av + w_ang*sc_ang;
  lkxx(0,0,size(3,3)) += mat33().eye()*w_av;
  lkx.head(3) += wk*w_av;
  lkxx(redang0index(),redang0index(),size(3,3)) += w_ang*dd_sc_ang;
  lkx(span(redang0index(),redang0index()+2)) += w_ang*d_sc_ang;
  double cross_cost = -sign(ddot)*as_scalar(ek.t()*Wq*wk)*w_avang;// w_avang*dot(wk,angerrvec);
  lkx.head(3) += -sign(ddot)*(ek.t()*Wq).t()*w_avang;
  lkx(span(redang0index(),redang0index()+2)) +=  -sign(ddot)*(ek.t()*join_rows(join_cols(vec({0.0}),wk),join_cols(-wk.t(),-skewSymmetric(wk)))*Wq).t()*w_avang;
  lkxx(0,redang0index(),size(3,3)) += sign(ddot)*We.t()*Wq*w_avang;
  lkxx(redang0index(),0,size(3,3)) += sign(ddot)*Wq.t()*We*w_avang;
  lkxx(redang0index(),redang0index(),size(3,3)) += mat33().eye()*sign(ddot)*as_scalar(ek.t()*Wq*wk)*w_avang;


  double actuation_cost = 0.0;
  double ang_mom_cost = 0.0;
  double stiction_cost = 0.0;

  lkuu += act_cost_mat*w_u_mult;
  if (useRawControlCost==0){
    lku += act_cost_mat*(uk-ukprev)*w_u_mult;
  }else{
    lku += act_cost_mat*(uk)*w_u_mult;
  }
  double state_mag_cost = w_avmag*abs(dot(wk,nb));
  double savang = sign(dot(wk,nb));
  lkx.head(3) += w_avmag*savang*nb;
  lkx(span(redang0index(),redang0index()+2)) += w_avmag*savang*(wk.t()*dBdq).t();
  lkxx(0,redang0index(),size(3,3)) += w_avmag*savang*dBdq;
  lkxx(redang0index(),redang0index(),size(3,3)) += w_avmag*savang*ddBwdq;
  lkxx(redang0index(),0,size(3,3)) += w_avmag*savang*dBdq.t();


  double act_mag_cost = 0.0;

  // RW costs using helper function
  RWCostResults rwCosts = computeRWCostsAndJacobians(xk, lkx, lkxx);
  ang_mom_cost += rwCosts.ang_mom_cost;
  stiction_cost += rwCosts.stiction_cost;

  cost_jacs out;
  out.lx = nj.t()*lkx;
  out.lxx = nj.t()*lkxx*nj;
  out.lux = lkux*nj;
  out.lu = lku;
  out.luu = lkuu;
  return out;
}

cost_jacs  Satellite::costJacobians(int k, int N, vec xk, vec uk,vec ukprev, vec3 satvec_k, vec ECIvec_k,vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const
{
  ExtractedCostSettings settings = ExtractedCostSettings::fromTuple(*costSettings_ptr);

  // Setup actuation cost matrix (handles terminal step weight update)
  mat act_cost_mat = setupActuationCostMatrix(k, N, settings);
  vec3 magvec = vec(3).zeros();
  if (k < N-1 && number_MTQ > 0) {
    magvec = mtq_ax_mat*uk.head(number_MTQ);
  }

  // Extract local copies of weights (may have been updated by setupActuationCostMatrix for terminal step)
  double w_ang = settings.w_ang;
  double w_av = settings.w_av;
  double w_u_mult = settings.w_u_mult;
  double w_avmag = settings.w_avmag;
  double w_avang = settings.w_avang;
  int considerVectorInTVLQR = settings.whichAngCostFunc;
  int useRawControlCost = settings.useRawControlCost;

  vec lkx = vec(reduced_state_N()).zeros();
  mat lkxx = mat(reduced_state_N(),reduced_state_N()).zeros();
  mat lkux = mat(control_N(),reduced_state_N()).zeros();
  vec lku = vec(control_N()).zeros();
  mat lkuu = mat(control_N(),control_N()).zeros();

  // xk = state_norm(xk);

  // vec4 qk = normalise(xk.rows(quat0index(), quat0index()+3));
  vec4 qk = xk.rows(quat0index(), quat0index()+3);
  vec3 wk = xk.rows(avindex0(),avindex0()+2);
  //vec v = satAlignVector;//get<1>(vu);
  vec3 sk = normalise(satvec_k);
  // vec ek = normalise(ECIvec_k);


  mat::fixed<4,3> Wq = findWMat(qk);
  // double ddot = norm_dot(sk,rotMat(qk).t()*ek);
  // tuple<vec4,vec4> xy = quatSetBasis(sk,ek);
  // vec4 x = get<0>(xy);
  // vec4 y = get<1>(xy);
  // vec4 p = closestQuat(qk,x,y);

  // double zz = dot(qk,p);
  // double sg = sign(zz);
  // zz *= sg;
  // double sc = 1-zz;

  // double phi = 2*acos(zz);
  // vec3 dphi = -2*(1.0/pow(1-zz*zz,0.5))*sg*Wq.t()*p;

  // vec4 dzdq = p;
  // double denom = sqrt(1-zz*zz);
  // vec3 dsc = -sg*Wq.t()*p;
  // mat44 ddzdq = (1/zz)*(x*trans(x) + y*trans(y) - p*trans(p));
  // mat33 ddsc = -sg*Wq.t()*ddzdq*Wq + mat33().eye()*zz;
  // ddsc = ddsc*cost_hess_mult;

  if(ECIvec_k.has_nan())
  {
    w_ang = 0.0;
  }

  vec3 nb = rotMat(qk).t()*normalise(BECI_k);
  mat::fixed<3,3> dBdq = dRTBdqQ(qk,normalise(BECI_k));
  mat::fixed<3,3> ddBwdq = ddvTRTudqQ(qk,wk,normalise(BECI_k));
  // mat::fixed<3,3> ddBmdq = ddvTRTudqQ(qk,magvec,normalise(BECI_k));
  cube::fixed<3,3,3> ddBdq = ddRTudqQ(qk,normalise(BECI_k));


  mat33 lkxx_quat_add = mat33().eye();//
  vec3 lkx_quat_add = vec(3).zeros();//
  vec3 angerrvec = vec(3).zeros();

  if(considerVectorInTVLQR==1){
      vec ek = normalise(ECIvec_k);
      if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){

        vec3 angerrvec = (cross(rotMat(qk).t()*ek,sk));
        ek = ek.tail(3);

        lkxx_quat_add = 0.5*(lkxx_quat_add - ddvTRTudqQ(qk,sk,ek));//
        lkx_quat_add = 0.5*(lkx_quat_add - (sk.t()*dRTBdqQ(qk,ek)).t() );
      }
    }
  // double state_cost = 0.5*as_scalar(wk.t()*wk*w_av) + sc;
  lkxx(0,0,size(3,3)) += mat33().eye()*w_av;
  lkx.head(3) += wk*w_av;
  lkxx(redang0index(),redang0index(),size(3,3)) += lkxx_quat_add*w_ang;
  lkx(span(redang0index(),redang0index()+2)) += w_ang*lkx_quat_add;
  // double cross_cost = w_avang*mat33().eye();
  lkx.head(3) += angerrvec*w_avang;
  lkx(span(redang0index(),redang0index()+2)) += w_avang*wk;
  lkxx(0,redang0index(),size(3,3)) += w_avang*mat33().eye();
  lkxx(redang0index(),0,size(3,3)) += w_avang*mat33().eye();


  // lkx.head(3) += -sign(ddot)*(ek.t()*Wq).t()*w_avang;
  // lkx(span(redang0index(),redang0index()+2)) +=  -sign(ddot)*(ek.t()*join_rows(join_cols(vec({0.0}),wk),join_cols(-wk.t(),-skewSymmetric(wk)))*Wq).t()*w_avang;
  // lkxx(0,redang0index(),size(3,3)) += sign(ddot)*We.t()*Wq*w_avang;
  // lkxx(redang0index(),0,size(3,3)) += sign(ddot)*Wq.t()*We*w_avang;
  // lkxx(redang0index(),redang0index(),size(3,3)) += mat33().eye()*sign(ddot)*as_scalar(ek.t()*Wq*wk)*w_avang;


  double actuation_cost = 0.0;
  double ang_mom_cost = 0.0;
  double stiction_cost = 0.0;

  if(k>0){
    if (useRawControlCost==0){
      lku += act_cost_mat*(uk-ukprev)*w_u_mult;
    }else{
      lku += act_cost_mat*(uk)*w_u_mult;
    }
    lkuu += act_cost_mat*w_u_mult;
  }else{
    if (useRawControlCost!=0){
      lku += act_cost_mat*(uk)*w_u_mult;
      lkuu += act_cost_mat*w_u_mult;
    }
  }
  double state_mag_cost = w_avmag*dot(wk,nb);
  double savang = sign(dot(wk,nb));
  lkx.head(3) += w_avmag*savang*nb;
  lkx(span(redang0index(),redang0index()+2)) += w_avmag*savang*(wk.t()*dBdq).t();
  lkxx(0,redang0index(),size(3,3)) += w_avmag*savang*dBdq;
  lkxx(redang0index(),redang0index(),size(3,3)) += w_avmag*savang*ddBwdq;
  lkxx(redang0index(),0,size(3,3)) += w_avmag*savang*dBdq.t();

  double act_mag_cost = 0.0;

  // RW costs using helper function (also fixes bug: stiction was using number_MTQ+j instead of 6+j)
  RWCostResults rwCosts = computeRWCostsAndJacobians(xk, lkx, lkxx);
  ang_mom_cost += rwCosts.ang_mom_cost;
  stiction_cost += rwCosts.stiction_cost;

  cost_jacs out;
  out.lx = lkx;
  out.lxx = lkxx;
  out.lux = lkux;
  out.lu = lku;
  out.luu = lkuu;
  return out;
}


double Satellite::read_magrw_torq_mult(){
  return MAGRW_TORQ_MULT;
}


/*This function gets the Imu matrix given mu, the ck vector, and lambda vector
  Arguments:
   double mu
   6 x 1 vec ck the constraints vector
   6 x 1 vec lamk the lambda vector
  Returns:
   6x6 mat the Imu matrix
*/
mat Satellite::getImu(double mu, vec muk, vec ck, vec lamk)
{
  mat Imu = diagmat(muk);//mat(constraint_N(),constraint_N()).eye()*mu;
  //mat Imu = diagmat(muk);

  for(int i = 0; i < ineq_constraint_N(); i++)
  {
    if(lamk(i)<=0 && ck(i)<=0)
    // if(lamk(i)<=0.0 && ck(i)<0.0)
    //if(ck(i)<0)
    {
      Imu(i,i) = 0.0;
    }
  }
  return Imu;
}
mat Satellite::getIlam(double mu, vec muk, vec ck, vec lamk)
{
  mat Ilam = mat(constraint_N(),constraint_N()).eye();
  return Ilam;
}




/* This function returns xdot according to BeaverCube's dynamics
   Arguments:
    int k = time
    u = dipole, 3 x 1
    x = state, 7 x 1
  Returns:
    xdot = state derivative, 7 x 1
*/

vec Satellite::dynamics_pure(vec x, vec u, DYNAMICS_INFO_FORM dynamics_info) const
{
  tuple<vec,vec3> dynout = dynamics(x,u,dynamics_info);
  vec out = get<0>(dynout);
  return out;
}

vec3 Satellite::dist_torque(vec x, DYNAMICS_INFO_FORM dynamics_info) const{

  vec3 Bk = get<0>(dynamics_info);
  vec3 Rk = get<1>(dynamics_info);
  int prop_torq_on = get<2>(dynamics_info);
  vec3 Vk = get<3>(dynamics_info);
  vec3 Sk = get<4>(dynamics_info);
  int dist_on = get<5>(dynamics_info);
  double rho = get<6>(dynamics_info);  // atmospheric density (kg/m^3)

  vec4 q = x.rows(quat0index(),quat0index()+3);
  mat33 RmatT = rotMat(q).t();

  // Gravity gradient torque
  double rad = norm(Rk);
  double const_term = plan_for_gg*3.0*earth_mu/(pow(rad*1000.0,3.0));
  vec3 nadir = -RmatT*Rk/rad;
  vec3 gg_torq = const_term*cross(nadir,Jcom*nadir);

  // Variable disturbances (propellant + general)
  vec3 variable_dist_torq = prop_torq_on*prop_torq*plan_for_prop+gen_dist_torq*plan_for_gendist;
  vec3 dist_torq = gg_torq + variable_dist_torq;

  // Residual dipole torque
  vec3 magvec = res_dipole*plan_for_resdipole;
  dist_torq += cross(magvec,RmatT*Bk);

  // Solar Radiation Pressure (SRP) torque using surface geometry
  // Based on Python model: srp_disturbance.py
  // T_srp = -P_solar * sum_i [ m_s,i * (r_i - COM) x s_body + m_n,i * (r_i - COM) x n_i ]
  // where m_s,i = A_i * (eta_a + eta_d) * cos_gamma_i
  //       m_n,i = A_i * (2*eta_s*cos_gamma_i^2 + 2/3*eta_d) * cos_gamma_i  (only if cos_gamma > 0)
  if(plan_for_srp && num_srp_surfaces > 0){
    // Sun direction in body frame (from satellite to sun)
    vec3 sun_dir_eci = normalise(Sk - Rk);  // ECI sun direction
    vec3 s_body = RmatT * sun_dir_eci;      // Body frame sun direction

    vec3 srp_torq = vec3().zeros();
    for(int i = 0; i < num_srp_surfaces; i++){
      vec3 n_i = srp_normals.col(i);
      vec3 r_i = srp_centroids.col(i) - srp_COM;  // lever arm from COM
      double A_i = srp_areas(i);
      double eta_s_i = srp_eta_s(i);
      double eta_d_i = srp_eta_d(i);
      double eta_a_i = srp_eta_a(i);

      double cos_gamma = dot(n_i, s_body);
      if(cos_gamma > 0){
        // Illuminated surface
        double m_s = A_i * (eta_a_i + eta_d_i) * cos_gamma;
        double m_n = A_i * (2.0*eta_s_i*cos_gamma*cos_gamma + (2.0/3.0)*eta_d_i) * cos_gamma;

        srp_torq += m_s * cross(r_i, s_body) + m_n * cross(r_i, n_i);
      }
    }
    dist_torq -= SOLAR_PRESSURE * srp_torq;
  }

  // Aerodynamic drag torque using surface geometry
  // Based on Python model: drag_disturbance.py
  // T_drag = -0.5 * rho * sum_i [ C_D,i * A_i * (n_i . V_body) * (r_i - COM) x V_body ]
  if(plan_for_aero && num_drag_surfaces > 0){
    // Velocity in body frame (Vk is in km/s, convert to m/s)
    vec3 V_body = RmatT * Vk * 1000.0;  // m/s
    double V_mag = norm(V_body);

    if(V_mag > 1e-6){
      vec3 drag_torq = vec3().zeros();
      for(int i = 0; i < num_drag_surfaces; i++){
        vec3 n_i = drag_normals.col(i);
        vec3 r_i = drag_centroids.col(i) - drag_COM;  // lever arm from COM
        double A_i = drag_areas(i);
        double CD_i = drag_CDs(i);

        double v_proj = dot(n_i, V_body);  // velocity projection onto surface normal
        if(v_proj > 0){
          // Surface facing the flow
          double F_i = CD_i * A_i * v_proj;
          drag_torq += F_i * cross(r_i, V_body);
        }
      }
      // rho is atmospheric density in kg/m^3
      dist_torq -= 0.5 * rho * drag_torq;
    }
  }

  return dist_torq*dist_on;
}

tuple<vec,vec3> Satellite::dynamics(vec x, vec u, DYNAMICS_INFO_FORM dynamics_info) const
{

  vec3 Bk = get<0>(dynamics_info);
  // vec3 Rk = get<1>(dynamics_info);
  // int prop_torq_on = get<2>(dynamics_info);
  // vec3 Vk = get<3>(dynamics_info);
  // vec3 Sk = get<4>(dynamics_info);
  // int dist_on = get<5>(dynamics_info);

  //vec3 prop_torq = this->prop_torq;
  //Find w, q
  vec3 w = x.rows(avindex0(),avindex0()+2);
  vec4 q = x.rows(quat0index(),quat0index()+3);
  vec h = vec(number_RW).zeros();
  vec3 rw_torq = vec3().zeros();
  if(number_RW>0){
    h = x.tail(number_RW);
    rw_torq = rw_ax_mat*u(span(number_MTQ,number_MTQ+number_RW-1));
  }
  vec3 magic_torq = vec3().zeros();
  if(number_magic>0){
    magic_torq = magic_ax_mat*u.tail(number_magic);
  }
  mat33 RmatT = rotMat(q).t();
  vec3 magvec = vec(3).zeros();
  if(number_MTQ>0)
  {
    magvec += mtq_ax_mat*u.head(number_MTQ);
  }
  vec3 torq = MAGRW_TORQ_MULT*(magic_torq + rw_torq);
  vec3 dist_torq = dist_torque(x,dynamics_info);


  //
  vec3 wdot = invJcom_noRW*(torq - cross(w, Jcom*w + rw_ax_mat*h) + cross(magvec,RmatT*Bk)  + dist_torq);
  vec res = join_cols(wdot,0.5*findWMat(q)*w);
  if(number_RW>0){
    res = join_cols(res,-MAGRW_TORQ_MULT*u(span(number_MTQ,number_MTQ+number_RW-1)) - diagmat(vec(RW_J))*rw_ax_mat.t()*wdot);
  }
  return std::make_tuple(res,dist_torq);
}


/* This function finds the Jacobian of BeaverCube's dynamics wrt x and u
   Arguments:
    state vector x (7 x 1)
    dipole vector u (3 x 1)
    magfield vector B, 3 x 1
    int t = time
  Returns:
    jxx - jacobian wrt x, 7 x 7
    jxu - jacobian wrt u, 7 x 3
*/
tuple<mat, mat,mat>  Satellite::dynamicsJacobians( vec x,  vec u,  DYNAMICS_INFO_FORM dynamics_info) const
{

  vec3 Bk = get<0>(dynamics_info);
  vec3 Rk = get<1>(dynamics_info);
  int prop_torq_on = get<2>(dynamics_info);
  vec3 Vk = get<3>(dynamics_info);
  vec3 Sk = get<4>(dynamics_info);
  int dist_on = get<5>(dynamics_info);

  vec3 w = x.rows(avindex0(),avindex0()+2);
  vec4 q = x.rows(quat0index(),quat0index()+3);
  vec h = vec(number_RW).zeros();
  vec3 rw_torq = vec3().zeros();
  if(number_RW>0){
    h = x.tail(number_RW);
    rw_torq = rw_ax_mat*u(span(number_MTQ,number_MTQ+number_RW-1));
  }
  vec3 magic_torq = vec3().zeros();
  if(number_magic>0){
    magic_torq = magic_ax_mat*u.tail(number_magic);
  }

  double const_term = dist_on*plan_for_gg*3.0*earth_mu/(pow(norm(Rk)*1000.0,3.0));
  vec3 nRk = normalise(Rk);
  mat33 RmatT = rotMat(q).t();
  vec3 nadir = -RmatT*nRk;
  vec3 gg_torq = const_term*cross(nadir,Jcom*nadir);


  vec3 magvec = dist_on*res_dipole*plan_for_resdipole;
  if(number_MTQ>0)
  {
    magvec += mtq_ax_mat*u.head(number_MTQ);
  }
  vec3 torq = MAGRW_TORQ_MULT*(magic_torq + rw_torq);
  vec3 dist_torq =  dist_on*(gg_torq+prop_torq_on*prop_torq*plan_for_prop+gen_dist_torq*plan_for_gendist);

  vec3 wdot = invJcom_noRW*(torq - cross(w, Jcom*w + rw_ax_mat*h) + cross(magvec,RmatT*Bk) + dist_torq);
  vec4 qdot = 0.5*findWMat(q)*w;
  vec hdot = vec(number_RW).zeros();
  if(number_RW>0)
  {
    hdot = -MAGRW_TORQ_MULT*u(span(number_MTQ,number_MTQ+number_RW-1)) - diagmat(vec(RW_J))*rw_ax_mat.t()*wdot;
  }

  mat33 dwdw = invJcom_noRW*( skewSymmetric(Jcom*w + rw_ax_mat*h)-skewSymmetric(w)*Jcom);
  mat::fixed<3,4> dwdq = invJcom_noRW*(skewSymmetric(magvec)*dRTBdq(q, Bk) + const_term*(skewSymmetric(nadir)*Jcom-skewSymmetric(Jcom*nadir))*dRTBdq(q, -nRk));

  // ===== SRP Jacobian =====
  if(dist_on && plan_for_srp && num_srp_surfaces > 0){
    vec3 sun_dir_eci = normalise(Sk - Rk);
    vec3 s_body = RmatT * sun_dir_eci;
    mat::fixed<3,4> ds_body_dq = dRTBdq(q, sun_dir_eci);

    mat::fixed<3,4> dT_srp_dq = mat(3,4).zeros();

    for(int i = 0; i < num_srp_surfaces; i++){
      vec3 n_i = srp_normals.col(i);
      vec3 r_i = srp_centroids.col(i) - srp_COM;
      double A_i = srp_areas(i);
      double eta_s_i = srp_eta_s(i);
      double eta_d_i = srp_eta_d(i);
      double eta_a_i = srp_eta_a(i);

      double cos_gamma = dot(n_i, s_body);
      if(cos_gamma > 0){
        rowvec4 dcos_gamma_dq = n_i.t() * ds_body_dq;

        double m_s = A_i * (eta_a_i + eta_d_i) * cos_gamma;
        rowvec4 dm_s_dq = A_i * (eta_a_i + eta_d_i) * dcos_gamma_dq;

        double dm_n_dcos = A_i * (6.0*eta_s_i*cos_gamma*cos_gamma + (2.0/3.0)*eta_d_i);
        rowvec4 dm_n_dq = dm_n_dcos * dcos_gamma_dq;

        vec3 r_cross_s = cross(r_i, s_body);
        vec3 r_cross_n = cross(r_i, n_i);
        mat::fixed<3,4> d_r_cross_s_dq = -skewSymmetric(r_i) * ds_body_dq;

        dT_srp_dq += r_cross_s * dm_s_dq + m_s * d_r_cross_s_dq + r_cross_n * dm_n_dq;
      }
    }
    dwdq -= SOLAR_PRESSURE * invJcom_noRW * dT_srp_dq;
  }

  // ===== Drag Jacobian =====
  if(dist_on && plan_for_aero && num_drag_surfaces > 0){
    double rho = get<6>(dynamics_info);
    vec3 Vk_scaled = Vk * 1000.0;
    vec3 V_body = RmatT * Vk_scaled;

    if(norm(V_body) > 1e-6){
      mat::fixed<3,4> dV_body_dq = dRTBdq(q, Vk_scaled);

      mat::fixed<3,4> dT_drag_dq = mat(3,4).zeros();

      for(int i = 0; i < num_drag_surfaces; i++){
        vec3 n_i = drag_normals.col(i);
        vec3 r_i = drag_centroids.col(i) - drag_COM;
        double A_i = drag_areas(i);
        double CD_i = drag_CDs(i);

        double v_proj = dot(n_i, V_body);
        if(v_proj > 0){
          rowvec4 dv_proj_dq = n_i.t() * dV_body_dq;

          double F_i = CD_i * A_i * v_proj;
          rowvec4 dF_i_dq = CD_i * A_i * dv_proj_dq;

          vec3 r_cross_V = cross(r_i, V_body);
          mat::fixed<3,4> d_r_cross_V_dq = -skewSymmetric(r_i) * dV_body_dq;

          dT_drag_dq += r_cross_V * dF_i_dq + F_i * d_r_cross_V_dq;
        }
      }
      dwdq -= 0.5 * rho * invJcom_noRW * dT_drag_dq;
    }
  }

  mat dwdh = invJcom_noRW*( -skewSymmetric(w)*rw_ax_mat);
  mat dwdu = invJcom_noRW*join_rows(-skewSymmetric(RmatT*Bk)*mtq_ax_mat,MAGRW_TORQ_MULT*rw_ax_mat, MAGRW_TORQ_MULT*magic_ax_mat);
  mat33 dwdt = invJcom_noRW;


  mat::fixed<4,3> dqdw = 0.5*findWMat(q);
  mat44 dqdq = 0.5*join_rows(join_cols(vec({0.0}),w),join_cols(-w.t(),-skewSymmetric(w)));
  mat dqdh = mat(4,number_RW).zeros();
  mat dqdu = mat(4,control_N()).zeros();
  mat::fixed<4,3> dqdt = mat(4,3).zeros();


  mat dhdw = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdw;
  mat dhdq =  -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdq;
  mat dhdh = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdh;
  mat dhdu = join_rows(mat(number_RW,number_MTQ).zeros(),-MAGRW_TORQ_MULT*mat(number_RW,number_RW).eye(),mat(number_RW,number_magic).zeros()) - diagmat(vec(RW_J))*rw_ax_mat.t()*dwdu;
  mat dhdt = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdt;


  mat jxx_augment = join_cols( join_rows(dwdw,dwdq,dwdh),
                                join_rows(dqdw,dqdq,dqdh));

  mat jxu_augment = join_cols(dwdu,dqdu);
  mat jxt_augment = join_cols(dwdt,dqdt);

  if(number_RW>0){
    jxx_augment = join_cols(jxx_augment,join_rows(dhdw,dhdq,dhdh));
    jxu_augment = join_cols(jxu_augment,dhdu);
    jxt_augment = join_cols(jxt_augment,dhdt);
  }

  return std::make_tuple(jxx_augment, jxu_augment,jxt_augment);

}



tuple<cube, cube,cube>  Satellite::dynamicsHessians( vec x,  vec u,  DYNAMICS_INFO_FORM dynamics_info) const
{

  vec3 Bk = get<0>(dynamics_info);
  vec3 Rk = get<1>(dynamics_info);
  int prop_torq_on = get<2>(dynamics_info);
  vec3 Vk = get<3>(dynamics_info);
  vec3 Sk = get<4>(dynamics_info);
  int dist_on = get<5>(dynamics_info);

  vec3 w = x.rows(avindex0(),avindex0()+2);
  vec4 q = x.rows(quat0index(),quat0index()+3);
  vec h = vec(number_RW).zeros();
  vec3 rw_torq = vec3().zeros();

  cube ddxd__dxdx = cube(state_N(),state_N(),state_N()).zeros();
  cube ddxd__dudx = cube(control_N(),state_N(),state_N()).zeros();
  cube ddxd__dudu = cube(control_N(),control_N(),state_N()).zeros();
  if(number_RW>0){
    h = x.tail(number_RW);
    rw_torq = rw_ax_mat*u(span(number_MTQ,number_MTQ+number_RW-1));
  }
  vec3 magic_torq = vec3().zeros();
  if(number_magic>0){
    magic_torq = magic_ax_mat*u.tail(number_magic);
  }

  double const_term = dist_on*plan_for_gg*3.0*earth_mu/(pow(norm(Rk)*1000.0,3.0));
  vec3 nRk = normalise(Rk);
  mat33 RmatT = rotMat(q).t();
  vec3 nadir = -RmatT*nRk;
  vec3 gg_torq = const_term*cross(nadir,Jcom*nadir);


  vec3 magvec = dist_on*res_dipole*plan_for_resdipole;
  if(number_MTQ>0)
  {
    magvec += mtq_ax_mat*u.head(number_MTQ);
  }
  vec3 torq = MAGRW_TORQ_MULT*(magic_torq + rw_torq);
  vec3 dist_torq =  dist_on*(gg_torq+prop_torq_on*prop_torq*plan_for_prop+gen_dist_torq*plan_for_gendist);

  vec3 wdot = invJcom_noRW*(torq - cross(w, Jcom*w + rw_ax_mat*h) + cross(magvec,RmatT*Bk) + dist_torq);
  vec4 qdot = 0.5*findWMat(q)*w;
  vec hdot = vec(number_RW).zeros();
  if(number_RW>0)
  {
    hdot = -MAGRW_TORQ_MULT*u(span(number_MTQ,number_MTQ+number_RW-1)) - diagmat(vec(RW_J))*rw_ax_mat.t()*wdot;
  }

  // mat33 dwdw = invJcom_noRW*( skewSymmetric(Jcom*w + rw_ax_mat*h)-skewSymmetric(w)*Jcom);
  // mat::fixed<3,4> dwdq = invJcom_noRW*(skewSymmetric(magvec)*dRTBdq(q, Bk) + const_term*(skewSymmetric(nadir)*Jcom-skewSymmetric(Jcom*nadir))*dRTBdq(q, -nRk));
  // mat dwdh = invJcom_noRW*( -skewSymmetric(w)*rw_ax_mat);
  // mat dwdu = invJcom_noRW*join_rows(-skewSymmetric(RmatT*Bk)*mtq_ax_mat,MAGRW_TORQ_MULT*rw_ax_mat, MAGRW_TORQ_MULT*magic_ax_mat);
  // mat33 dwdt = invJcom_noRW;

  vec3 xh = mat33().eye().col(0);
  vec3 yh = mat33().eye().col(1);
  vec3 zh = mat33().eye().col(2);


  // a.t()*ss(b) = -a.t()*ss(b).t() = -(ss(b)*a).t() = -cross(b,a).t() = cross(a,b).t()
  // cross(invJcom_noRW.t()*xh,Jcom*w + rw_ax_mat*h).t()-cross(invJcom_noRW.t()*xh,w).t()*Jcom;
  // (Jcom*w + rw_ax_mat*h).t()*skewSymmetric(invJcom_noRW.t()*xh).t()-w.t()*skewSymmetric(invJcom_noRW.t()*xh).t()*Jcom;
  ddxd__dxdx.slice(0)(span(avindex0(),avindex0()+2),span(avindex0(),avindex0()+2)) =   Jcom.t()*skewSymmetric(invJcom_noRW.t()*xh).t()-skewSymmetric(invJcom_noRW.t()*xh).t()*Jcom;
  ddxd__dxdx.slice(1)(span(avindex0(),avindex0()+2),span(avindex0(),avindex0()+2)) =   Jcom.t()*skewSymmetric(invJcom_noRW.t()*yh).t()-skewSymmetric(invJcom_noRW.t()*yh).t()*Jcom;
  ddxd__dxdx.slice(2)(span(avindex0(),avindex0()+2),span(avindex0(),avindex0()+2)) =   Jcom.t()*skewSymmetric(invJcom_noRW.t()*zh).t()-skewSymmetric(invJcom_noRW.t()*zh).t()*Jcom;


  // mat::fixed<3,4> dwdq = invJcom_noRW*(skewSymmetric(magvec)*dRTBdq(q, Bk) + const_term*(skewSymmetric(nadir)*Jcom-skewSymmetric(Jcom*nadir))*dRTBdq(q, -nRk));
  // ddvTRTudqQ(q,cross(invJcom_noRW.t()*xh,magvec),Bk) + const_term*ddvTRTudqQ(q,Jcom*cross(invJcom_noRW.t()*xh,nadir)-cross(invJcom_noRW.t()*xh,Jcom*nadir), -nRk);

  ddxd__dxdx.slice(0)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) =  ddvTRTudq(q,cross(invJcom_noRW.t()*xh,magvec),Bk) + const_term*ddvTRTudq(q,Jcom*cross(invJcom_noRW.t()*xh,nadir)-cross(invJcom_noRW.t()*xh,Jcom*nadir), -nRk);
  ddxd__dxdx.slice(1)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) =  ddvTRTudq(q,cross(invJcom_noRW.t()*yh,magvec),Bk) + const_term*ddvTRTudq(q,Jcom*cross(invJcom_noRW.t()*yh,nadir)-cross(invJcom_noRW.t()*yh,Jcom*nadir), -nRk);
  ddxd__dxdx.slice(2)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3 )) = ddvTRTudq(q,cross(invJcom_noRW.t()*zh,magvec),Bk) + const_term*ddvTRTudq(q,Jcom*cross(invJcom_noRW.t()*zh,nadir)-cross(invJcom_noRW.t()*zh,Jcom*nadir), -nRk);

  // ===== SRP Hessian =====
  // The SRP torque depends on quaternion through: s_body = R^T(q) * sun_dir_eci
  // For each wdot component, we need d^2(wdot_i)/dq^2
  // wdot = invJcom * T_srp, so d^2(wdot)/dq^2 involves d^2(T_srp)/dq^2
  // T_srp_i = m_s_i * (r_i x s_body) + m_n_i * (r_i x n_i)
  // The hessian follows the pattern from existing GG/magnetic: use ddvTRTudq for second derivatives
  if(dist_on && plan_for_srp && num_srp_surfaces > 0){
    vec3 sun_dir_eci = normalise(Sk - Rk);
    vec3 s_body = RmatT * sun_dir_eci;
    mat::fixed<3,4> ds_body_dq = dRTBdq(q, sun_dir_eci);

    // For each wdot component (0, 1, 2), compute the SRP hessian contribution
    vec3 basis_vecs[3] = {xh, yh, zh};
    for(int comp = 0; comp < 3; comp++){
      vec3 e_comp = basis_vecs[comp];
      vec3 invJ_e = invJcom_noRW.t() * e_comp;

      mat44 ddwdot_srp_dqdq = mat44().zeros();

      for(int i = 0; i < num_srp_surfaces; i++){
        vec3 n_i = srp_normals.col(i);
        vec3 r_i = srp_centroids.col(i) - srp_COM;
        double A_i = srp_areas(i);
        double eta_s_i = srp_eta_s(i);
        double eta_d_i = srp_eta_d(i);
        double eta_a_i = srp_eta_a(i);

        double cos_gamma = dot(n_i, s_body);
        if(cos_gamma > 0){
          // First derivatives (needed for product rule in hessian)
          rowvec4 dcos_gamma_dq = n_i.t() * ds_body_dq;

          double m_s = A_i * (eta_a_i + eta_d_i) * cos_gamma;
          rowvec4 dm_s_dq = A_i * (eta_a_i + eta_d_i) * dcos_gamma_dq;

          double m_n_coeff = 2.0*eta_s_i*cos_gamma*cos_gamma + (2.0/3.0)*eta_d_i;
          double m_n = A_i * m_n_coeff * cos_gamma;
          double dm_n_dcos = A_i * (6.0*eta_s_i*cos_gamma*cos_gamma + (2.0/3.0)*eta_d_i);
          rowvec4 dm_n_dq = dm_n_dcos * dcos_gamma_dq;

          vec3 r_cross_s = cross(r_i, s_body);
          vec3 r_cross_n = cross(r_i, n_i);

          // Second derivatives
          // d^2(cos_gamma)/dq^2 = d^2(n_i . R^T * sun)/dq^2 = ddvTRTudq(q, n_i, sun_dir_eci)
          mat44 ddcos_gamma_dqdq = ddvTRTudq(q, n_i, sun_dir_eci);

          // d^2(m_s)/dq^2 = A*(eta_a+eta_d) * d^2(cos_gamma)/dq^2
          mat44 ddm_s_dqdq = A_i * (eta_a_i + eta_d_i) * ddcos_gamma_dqdq;

          // d^2(m_n)/dq^2: m_n = A*(2*eta_s*cos^3 + (2/3)*eta_d*cos)
          // dm_n/dcos = A*(6*eta_s*cos^2 + (2/3)*eta_d)
          // d^2m_n/dcos^2 = A*12*eta_s*cos
          double ddm_n_dcos2 = A_i * 12.0 * eta_s_i * cos_gamma;
          mat44 ddm_n_dqdq = ddm_n_dcos2 * dcos_gamma_dq.t() * dcos_gamma_dq + dm_n_dcos * ddcos_gamma_dqdq;

          // d^2(r x s_body)/dq^2 = -skew(r) * d^2(s_body)/dq^2
          // For a fixed direction vector, d^2(R^T*u)/dq^2 projected onto direction v:
          // We need d^2(v . R^T * u)/dq^2 = ddvTRTudq(q, v, u)
          // For cross product: (r x s) component along e_comp = e_comp . (r x s) = (e_comp x r) . s
          // So d^2((r x s) . e_comp)/dq^2 = d^2((e_comp x r) . R^T * sun)/dq^2 = ddvTRTudq(q, cross(e_comp,r), sun)
          vec3 e_cross_r = cross(e_comp, r_i);
          mat44 dd_r_cross_s_comp_dqdq = ddvTRTudq(q, e_cross_r, sun_dir_eci);

          // The torque contribution to wdot_comp is: invJ_e . T_srp = invJ_e . (m_s*(r x s) + m_n*(r x n))
          // = m_s * (invJ_e . (r x s)) + m_n * (invJ_e . (r x n))
          // = m_s * ((invJ_e x r) . s) + m_n * (invJ_e . (r x n))
          // The second term is constant in q (n is fixed in body frame)
          // d^2/dq^2 of first term uses product rule:
          // d^2(m_s * v.s)/dq^2 = d^2m_s/dq^2 * (v.s) + 2*dm_s/dq * d(v.s)/dq^T + m_s * d^2(v.s)/dq^2
          vec3 invJ_e_cross_r = cross(invJ_e, r_i);
          double invJ_e_dot_r_cross_s = dot(invJ_e_cross_r, s_body);
          rowvec4 d_invJ_e_dot_r_cross_s_dq = invJ_e_cross_r.t() * ds_body_dq;
          mat44 dd_invJ_e_dot_r_cross_s_dqdq = ddvTRTudq(q, invJ_e_cross_r, sun_dir_eci);

          // Product rule for m_s * (invJ_e . (r x s))
          mat44 dd_m_s_term_dqdq = ddm_s_dqdq * invJ_e_dot_r_cross_s
                                 + dm_s_dq.t() * d_invJ_e_dot_r_cross_s_dq
                                 + d_invJ_e_dot_r_cross_s_dq.t() * dm_s_dq
                                 + m_s * dd_invJ_e_dot_r_cross_s_dqdq;

          // Product rule for m_n * (invJ_e . (r x n)) - but (r x n) is constant, so only m_n varies
          double invJ_e_dot_r_cross_n = dot(invJ_e, r_cross_n);
          mat44 dd_m_n_term_dqdq = ddm_n_dqdq * invJ_e_dot_r_cross_n;

          ddwdot_srp_dqdq += dd_m_s_term_dqdq + dd_m_n_term_dqdq;
        }
      }
      ddxd__dxdx.slice(comp)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) -= SOLAR_PRESSURE * ddwdot_srp_dqdq;
    }
  }

  // ===== Drag Hessian =====
  // Similar structure to SRP but simpler since F_i is linear in v_proj
  if(dist_on && plan_for_aero && num_drag_surfaces > 0){
    double rho = get<6>(dynamics_info);
    vec3 Vk_scaled = Vk * 1000.0;
    vec3 V_body = RmatT * Vk_scaled;

    if(norm(V_body) > 1e-6){
      mat::fixed<3,4> dV_body_dq = dRTBdq(q, Vk_scaled);

      vec3 basis_vecs[3] = {xh, yh, zh};
      for(int comp = 0; comp < 3; comp++){
        vec3 e_comp = basis_vecs[comp];
        vec3 invJ_e = invJcom_noRW.t() * e_comp;

        mat44 ddwdot_drag_dqdq = mat44().zeros();

        for(int i = 0; i < num_drag_surfaces; i++){
          vec3 n_i = drag_normals.col(i);
          vec3 r_i = drag_centroids.col(i) - drag_COM;
          double A_i = drag_areas(i);
          double CD_i = drag_CDs(i);

          double v_proj = dot(n_i, V_body);
          if(v_proj > 0){
            // First derivatives
            rowvec4 dv_proj_dq = n_i.t() * dV_body_dq;

            double F_i = CD_i * A_i * v_proj;
            rowvec4 dF_i_dq = CD_i * A_i * dv_proj_dq;

            vec3 r_cross_V = cross(r_i, V_body);

            // Second derivatives
            // d^2(v_proj)/dq^2 = d^2(n_i . R^T * Vk)/dq^2
            mat44 ddv_proj_dqdq = ddvTRTudq(q, n_i, Vk_scaled);

            // d^2(F_i)/dq^2 = CD*A * d^2(v_proj)/dq^2
            mat44 ddF_i_dqdq = CD_i * A_i * ddv_proj_dqdq;

            // The torque contribution: T_drag_i = F_i * (r x V)
            // wdot_comp contribution = invJ_e . T_drag_i = F_i * (invJ_e . (r x V))
            //                        = F_i * ((invJ_e x r) . V)
            vec3 invJ_e_cross_r = cross(invJ_e, r_i);
            double invJ_e_dot_r_cross_V = dot(invJ_e_cross_r, V_body);
            rowvec4 d_invJ_e_dot_r_cross_V_dq = invJ_e_cross_r.t() * dV_body_dq;
            mat44 dd_invJ_e_dot_r_cross_V_dqdq = ddvTRTudq(q, invJ_e_cross_r, Vk_scaled);

            // Product rule for F_i * (invJ_e . (r x V))
            mat44 dd_F_term_dqdq = ddF_i_dqdq * invJ_e_dot_r_cross_V
                                 + dF_i_dq.t() * d_invJ_e_dot_r_cross_V_dq
                                 + d_invJ_e_dot_r_cross_V_dq.t() * dF_i_dq
                                 + F_i * dd_invJ_e_dot_r_cross_V_dqdq;

            ddwdot_drag_dqdq += dd_F_term_dqdq;
          }
        }
        ddxd__dxdx.slice(comp)(span(quat0index(),quat0index()+3),span(quat0index(),quat0index()+3)) -= 0.5 * rho * ddwdot_drag_dqdq;
      }
    }
  }

  if(number_RW>0)
  {
    if(number_RW==1){
      ddxd__dxdx.slice(0)(span(quat0index()+4),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*xh).t();
      ddxd__dxdx.slice(1)(span(quat0index()+4),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*yh).t();
      ddxd__dxdx.slice(2)(span(quat0index()+4),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*zh).t();

      ddxd__dxdx.slice(0)(span(avindex0(),avindex0()+2),span(quat0index()+4)) =   skewSymmetric(invJcom_noRW.t()*xh)*(rw_ax_mat);
      ddxd__dxdx.slice(1)(span(avindex0(),avindex0()+2),span(quat0index()+4)) =   skewSymmetric(invJcom_noRW.t()*yh)*(rw_ax_mat);
      ddxd__dxdx.slice(2)(span(avindex0(),avindex0()+2),span(quat0index()+4)) =   skewSymmetric(invJcom_noRW.t()*zh)*(rw_ax_mat);
    }
    else
    {
      ddxd__dxdx.slice(0)(span(quat0index()+4,quat0index()+3+number_RW),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*xh).t();
      ddxd__dxdx.slice(1)(span(quat0index()+4,quat0index()+3+number_RW),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*yh).t();
      ddxd__dxdx.slice(2)(span(quat0index()+4,quat0index()+3+number_RW),span(avindex0(),avindex0()+2)) =   (rw_ax_mat).t()*skewSymmetric(invJcom_noRW.t()*zh).t();

      ddxd__dxdx.slice(0)(span(avindex0(),avindex0()+2),span(quat0index()+4,quat0index()+3+number_RW)) =   skewSymmetric(invJcom_noRW.t()*xh)*(rw_ax_mat);
      ddxd__dxdx.slice(1)(span(avindex0(),avindex0()+2),span(quat0index()+4,quat0index()+3+number_RW)) =   skewSymmetric(invJcom_noRW.t()*yh)*(rw_ax_mat);
      ddxd__dxdx.slice(2)(span(avindex0(),avindex0()+2),span(quat0index()+4,quat0index()+3+number_RW)) =   skewSymmetric(invJcom_noRW.t()*zh)*(rw_ax_mat);

    }
  }
  if(number_MTQ>0){
    ddxd__dudx.slice(0)(span(0,number_MTQ-1),span(quat0index(),quat0index()+3)) = mtq_ax_mat.t()*skewSymmetric(invJcom_noRW.t()*xh).t()*dRTBdq(q, Bk);
    ddxd__dudx.slice(1)(span(0,number_MTQ-1),span(quat0index(),quat0index()+3)) = mtq_ax_mat.t()*skewSymmetric(invJcom_noRW.t()*yh).t()*dRTBdq(q, Bk);
    ddxd__dudx.slice(2)(span(0,number_MTQ-1),span(quat0index(),quat0index()+3)) = mtq_ax_mat.t()*skewSymmetric(invJcom_noRW.t()*zh).t()*dRTBdq(q, Bk);
  }


  for(int ei = 0; ei<4;ei++){
      vec4 eiv = mat44().eye().col(ei);
      ddxd__dxdx.slice(quat0index()+ei)(span(avindex0(),avindex0()+2),span(quat0index(),quat0index()+3)) = -0.5*findWMat(eiv).t();
      ddxd__dxdx.slice(quat0index()+ei)(span(quat0index(),quat0index()+3),span(avindex0(),avindex0()+2)) = -0.5*findWMat(eiv);
  }

  if(number_RW>0)
  {
    for(int ei = 0; ei<number_RW;ei++){
      vec eiv = mat(number_RW,number_RW).eye().col(ei);
      for(int ei2 = 0; ei2<3;ei2++){
        vec3 eiv2 = mat(3,3).eye().col(ei2);
        ddxd__dxdx.slice(quat0index()+4+ei) += -RW_J.at(ei)*rw_ax_mat(ei2,ei)*ddxd__dxdx.slice(ei2);
        ddxd__dudx.slice(quat0index()+4+ei) += -as_scalar(eiv.t()*diagmat(vec(RW_J))*rw_ax_mat.t()*eiv2)*ddxd__dudx.slice(ei2);
        ddxd__dudu.slice(quat0index()+4+ei) += -as_scalar(eiv.t()*diagmat(vec(RW_J))*rw_ax_mat.t()*eiv2)*ddxd__dudu.slice(ei2);
      }
    }
  }

  return std::make_tuple(ddxd__dxdx, ddxd__dudx,ddxd__dudu);

}


/*
 * Blended dynamics for constraint tightening warm-start.
 *
 * Interpolates MTQ torque model between:
 *   alpha=0: tau = |B| * P_perp * m  (linear in m, scaled by field strength)
 *            where P_perp = I - Bhat*Bhat^T projects onto plane perp to B
 *   alpha=1: tau = m x B             (true cross-product physics)
 *
 * The alpha=0 formulation has a full-rank control Jacobian (no null space along B),
 * which aids optimization convergence in early iterations.
 *
 * Arguments:
 *   x - state vector [omega, q, h_rw]
 *   u - control vector [u_mtq, u_rw, u_magic]
 *   dynamics_info - tuple with B-field, position, etc.
 *   alpha - blending parameter in [0,1]
 *
 * Returns:
 *   tuple of (state_derivative, disturbance_torque)
 */
tuple<vec,vec3> Satellite::dynamicsBlended(vec x, vec u, DYNAMICS_INFO_FORM dynamics_info, double alpha) const
{
  vec3 Bk = get<0>(dynamics_info);

  // Extract state
  vec3 w = x.rows(avindex0(),avindex0()+2);
  vec4 q = x.rows(quat0index(),quat0index()+3);
  vec h = vec(number_RW).zeros();
  vec3 rw_torq = vec3().zeros();
  if(number_RW>0){
    h = x.tail(number_RW);
    rw_torq = rw_ax_mat*u(span(number_MTQ,number_MTQ+number_RW-1));
  }
  vec3 magic_torq = vec3().zeros();
  if(number_magic>0){
    magic_torq = magic_ax_mat*u.tail(number_magic);
  }

  mat33 RmatT = rotMat(q).t();
  vec3 B_body = RmatT*Bk;
  double B_norm = norm(B_body);

  // Compute MTQ dipole moment in body frame
  vec3 magvec = vec(3).zeros();
  if(number_MTQ>0)
  {
    magvec = mtq_ax_mat*u.head(number_MTQ);
  }

  // Compute MTQ torque with blending
  vec3 mtq_torq = vec(3).zeros();
  if(number_MTQ > 0 && B_norm > 1e-12){
    // True physics: tau = m x B
    vec3 tau_true = cross(magvec, B_body);

    // Relaxed physics: tau = |B| * P_perp * m
    // where P_perp = I - Bhat*Bhat^T
    vec3 Bhat = B_body / B_norm;
    vec3 m_perp = magvec - dot(magvec, Bhat) * Bhat;  // P_perp * m
    vec3 tau_relaxed = B_norm * m_perp;

    // Blend
    mtq_torq = alpha * tau_true + (1.0 - alpha) * tau_relaxed;
  }

  vec3 torq = MAGRW_TORQ_MULT*(magic_torq + rw_torq);
  vec3 dist_torq = dist_torque(x, dynamics_info);

  // Angular velocity derivative
  vec3 wdot = invJcom_noRW*(torq - cross(w, Jcom*w + rw_ax_mat*h) + mtq_torq + dist_torq);

  // Quaternion derivative
  vec res = join_cols(wdot, 0.5*findWMat(q)*w);

  // Reaction wheel momentum derivative
  if(number_RW>0){
    res = join_cols(res, -MAGRW_TORQ_MULT*u(span(number_MTQ,number_MTQ+number_RW-1))
                         - diagmat(vec(RW_J))*rw_ax_mat.t()*wdot);
  }

  return std::make_tuple(res, dist_torq);
}


/*
 * Blended dynamics Jacobians for constraint tightening warm-start.
 *
 * Control Jacobian (dwdu) interpolates between:
 *   alpha=0: d(wdot)/d(u_mtq) = J^{-1} * |B| * P_perp * mtq_ax_mat
 *            Full rank, well-conditioned
 *   alpha=1: d(wdot)/d(u_mtq) = -J^{-1} * [Bx] * mtq_ax_mat
 *            Rank 2 (null space along B), true physics
 *
 * State Jacobian (dwdq) also blended for consistency.
 *
 * Arguments:
 *   x - state vector
 *   u - control vector
 *   dynamics_info - B-field, position, etc.
 *   alpha - blending parameter in [0,1]
 *
 * Returns:
 *   tuple of (jxx, jxu, jxt) - state, control, and disturbance torque Jacobians
 */
tuple<mat, mat, mat> Satellite::dynamicsJacobiansBlended(vec x, vec u, DYNAMICS_INFO_FORM dynamics_info, double alpha) const
{
  vec3 Bk = get<0>(dynamics_info);
  vec3 Rk = get<1>(dynamics_info);
  int prop_torq_on = get<2>(dynamics_info);
  vec3 Vk = get<3>(dynamics_info);
  vec3 Sk = get<4>(dynamics_info);
  int dist_on = get<5>(dynamics_info);

  vec3 w = x.rows(avindex0(),avindex0()+2);
  vec4 q = x.rows(quat0index(),quat0index()+3);
  vec h = vec(number_RW).zeros();
  vec3 rw_torq = vec3().zeros();
  if(number_RW>0){
    h = x.tail(number_RW);
    rw_torq = rw_ax_mat*u(span(number_MTQ,number_MTQ+number_RW-1));
  }
  vec3 magic_torq = vec3().zeros();
  if(number_magic>0){
    magic_torq = magic_ax_mat*u.tail(number_magic);
  }

  double const_term = dist_on*plan_for_gg*3.0*earth_mu/(pow(norm(Rk)*1000.0,3.0));
  vec3 nRk = normalise(Rk);
  mat33 RmatT = rotMat(q).t();
  vec3 nadir = -RmatT*nRk;
  vec3 gg_torq = const_term*cross(nadir,Jcom*nadir);

  vec3 B_body = RmatT*Bk;
  double B_norm = norm(B_body);

  vec3 magvec = dist_on*res_dipole*plan_for_resdipole;
  if(number_MTQ>0)
  {
    magvec += mtq_ax_mat*u.head(number_MTQ);
  }
  vec3 torq = MAGRW_TORQ_MULT*(magic_torq + rw_torq);
  vec3 dist_torq = dist_on*(gg_torq+prop_torq_on*prop_torq*plan_for_prop+gen_dist_torq*plan_for_gendist);

  // ===== Blended MTQ torque for wdot computation =====
  vec3 mtq_torq = vec(3).zeros();
  if(number_MTQ > 0 && B_norm > 1e-12){
    vec3 tau_true = cross(magvec, B_body);
    vec3 Bhat = B_body / B_norm;
    vec3 m_perp = magvec - dot(magvec, Bhat) * Bhat;
    vec3 tau_relaxed = B_norm * m_perp;
    mtq_torq = alpha * tau_true + (1.0 - alpha) * tau_relaxed;
  }

  vec3 wdot = invJcom_noRW*(torq - cross(w, Jcom*w + rw_ax_mat*h) + mtq_torq + dist_torq);
  vec4 qdot = 0.5*findWMat(q)*w;
  vec hdot = vec(number_RW).zeros();
  if(number_RW>0)
  {
    hdot = -MAGRW_TORQ_MULT*u(span(number_MTQ,number_MTQ+number_RW-1)) - diagmat(vec(RW_J))*rw_ax_mat.t()*wdot;
  }

  // ===== State Jacobian (dwdw) - same as original =====
  mat33 dwdw = invJcom_noRW*(skewSymmetric(Jcom*w + rw_ax_mat*h) - skewSymmetric(w)*Jcom);

  // ===== State Jacobian (dwdq) - blended =====
  // True physics term: skewSymmetric(magvec) * dRTBdq(q, Bk)
  // Relaxed term: |B| * P_perp depends on q through B_body direction
  //   d(|B| * P_perp * m)/dq = |B| * d(P_perp)/dq * m
  //   Since P_perp = I - Bhat*Bhat^T and Bhat = R^T*B_ECI/|B|
  //   d(Bhat)/dq = dRTBdq(q, B_ECI)/|B|
  //   d(P_perp*m)/dq = -d(Bhat*(Bhat.m))/dq = -(d(Bhat)/dq * (Bhat.m) + Bhat * (d(Bhat)/dq)^T * m)

  mat::fixed<3,4> dwdq_mtq = mat(3,4).zeros();
  if(number_MTQ > 0 && B_norm > 1e-12){
    // True physics Jacobian
    mat::fixed<3,4> dwdq_true = skewSymmetric(magvec) * dRTBdq(q, Bk);

    // Relaxed physics Jacobian
    vec3 Bhat = B_body / B_norm;
    double m_dot_Bhat = dot(magvec, Bhat);
    mat::fixed<3,4> dBhat_dq = dRTBdq(q, Bk) / B_norm;

    // d(P_perp * m)/dq = -Bhat * (m^T * dBhat_dq) - dBhat_dq * (Bhat^T * m)
    mat::fixed<3,4> dPperp_m_dq = -Bhat * (magvec.t() * dBhat_dq) - dBhat_dq * m_dot_Bhat;
    mat::fixed<3,4> dwdq_relaxed = B_norm * dPperp_m_dq;

    dwdq_mtq = alpha * dwdq_true + (1.0 - alpha) * dwdq_relaxed;
  }

  mat::fixed<3,4> dwdq = invJcom_noRW * (dwdq_mtq + const_term*(skewSymmetric(nadir)*Jcom - skewSymmetric(Jcom*nadir))*dRTBdq(q, -nRk));

  // ===== SRP Jacobian (same as original) =====
  if(dist_on && plan_for_srp && num_srp_surfaces > 0){
    vec3 sun_dir_eci = normalise(Sk - Rk);
    vec3 s_body = RmatT * sun_dir_eci;
    mat::fixed<3,4> ds_body_dq = dRTBdq(q, sun_dir_eci);

    mat::fixed<3,4> dT_srp_dq = mat(3,4).zeros();

    for(int i = 0; i < num_srp_surfaces; i++){
      vec3 n_i = srp_normals.col(i);
      vec3 r_i = srp_centroids.col(i) - srp_COM;
      double A_i = srp_areas(i);
      double eta_s_i = srp_eta_s(i);
      double eta_d_i = srp_eta_d(i);
      double eta_a_i = srp_eta_a(i);

      double cos_gamma = dot(n_i, s_body);
      if(cos_gamma > 0){
        rowvec4 dcos_gamma_dq = n_i.t() * ds_body_dq;

        double m_s = A_i * (eta_a_i + eta_d_i) * cos_gamma;
        rowvec4 dm_s_dq = A_i * (eta_a_i + eta_d_i) * dcos_gamma_dq;

        double dm_n_dcos = A_i * (6.0*eta_s_i*cos_gamma*cos_gamma + (2.0/3.0)*eta_d_i);
        rowvec4 dm_n_dq = dm_n_dcos * dcos_gamma_dq;

        vec3 r_cross_s = cross(r_i, s_body);
        vec3 r_cross_n = cross(r_i, n_i);
        mat::fixed<3,4> d_r_cross_s_dq = -skewSymmetric(r_i) * ds_body_dq;

        dT_srp_dq += r_cross_s * dm_s_dq + m_s * d_r_cross_s_dq + r_cross_n * dm_n_dq;
      }
    }
    dwdq -= SOLAR_PRESSURE * invJcom_noRW * dT_srp_dq;
  }

  // ===== Drag Jacobian (same as original) =====
  if(dist_on && plan_for_aero && num_drag_surfaces > 0){
    double rho = get<6>(dynamics_info);
    vec3 Vk_scaled = Vk * 1000.0;
    vec3 V_body = RmatT * Vk_scaled;

    if(norm(V_body) > 1e-6){
      mat::fixed<3,4> dV_body_dq = dRTBdq(q, Vk_scaled);

      mat::fixed<3,4> dT_drag_dq = mat(3,4).zeros();

      for(int i = 0; i < num_drag_surfaces; i++){
        vec3 n_i = drag_normals.col(i);
        vec3 r_i = drag_centroids.col(i) - drag_COM;
        double A_i = drag_areas(i);
        double CD_i = drag_CDs(i);

        double v_proj = dot(n_i, V_body);
        if(v_proj > 0){
          rowvec4 dv_proj_dq = n_i.t() * dV_body_dq;

          double F_i = CD_i * A_i * v_proj;
          rowvec4 dF_i_dq = CD_i * A_i * dv_proj_dq;

          vec3 r_cross_V = cross(r_i, V_body);
          mat::fixed<3,4> d_r_cross_V_dq = -skewSymmetric(r_i) * dV_body_dq;

          dT_drag_dq += r_cross_V * dF_i_dq + F_i * d_r_cross_V_dq;
        }
      }
      dwdq -= 0.5 * rho * invJcom_noRW * dT_drag_dq;
    }
  }

  mat dwdh = invJcom_noRW * (-skewSymmetric(w) * rw_ax_mat);

  // ===== Control Jacobian (dwdu) - blended =====
  // True physics: -skewSymmetric(B_body) * mtq_ax_mat
  // Relaxed: |B| * P_perp * mtq_ax_mat = |B| * (I - Bhat*Bhat^T) * mtq_ax_mat
  mat dwdu_mtq = mat(3, number_MTQ).zeros();
  if(number_MTQ > 0 && B_norm > 1e-12){
    mat33 dwdu_true = -skewSymmetric(B_body);

    vec3 Bhat = B_body / B_norm;
    mat33 P_perp = mat33().eye() - Bhat * Bhat.t();
    mat33 dwdu_relaxed = B_norm * P_perp;

    mat33 dwdu_blended = alpha * dwdu_true + (1.0 - alpha) * dwdu_relaxed;
    dwdu_mtq = dwdu_blended * mtq_ax_mat;
  }

  mat dwdu = invJcom_noRW * join_rows(dwdu_mtq, MAGRW_TORQ_MULT*rw_ax_mat, MAGRW_TORQ_MULT*magic_ax_mat);
  mat33 dwdt = invJcom_noRW;

  mat::fixed<4,3> dqdw = 0.5*findWMat(q);
  mat44 dqdq = 0.5*join_rows(join_cols(vec({0.0}),w),join_cols(-w.t(),-skewSymmetric(w)));
  mat dqdh = mat(4,number_RW).zeros();
  mat dqdu = mat(4,control_N()).zeros();
  mat::fixed<4,3> dqdt = mat(4,3).zeros();

  mat dhdw = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdw;
  mat dhdq = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdq;
  mat dhdh = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdh;
  mat dhdu = join_rows(mat(number_RW,number_MTQ).zeros(),
                       -MAGRW_TORQ_MULT*mat(number_RW,number_RW).eye(),
                       mat(number_RW,number_magic).zeros())
             - diagmat(vec(RW_J))*rw_ax_mat.t()*dwdu;
  mat dhdt = -diagmat(vec(RW_J))*rw_ax_mat.t()*dwdt;

  mat jxx_augment = join_cols(join_rows(dwdw,dwdq,dwdh),
                              join_rows(dqdw,dqdq,dqdh));

  mat jxu_augment = join_cols(dwdu,dqdu);
  mat jxt_augment = join_cols(dwdt,dqdt);

  if(number_RW>0){
    jxx_augment = join_cols(jxx_augment,join_rows(dhdw,dhdq,dhdh));
    jxu_augment = join_cols(jxu_augment,dhdu);
    jxt_augment = join_cols(jxt_augment,dhdt);
  }

  return std::make_tuple(jxx_augment, jxu_augment, jxt_augment);
}




Satellite::~Satellite(){
  MTQ_axes.clear();
  RW_axes.clear();
  magic_axes.clear();
  MTQ_max.clear();
  magic_max_torq.clear();
  RW_max_torq.clear();
  RW_max_ang_mom.clear();
  MTQ_cost.clear();
  RW_cost.clear();
  magic_cost.clear();
  RW_AM_cost.clear();
  RW_AM_cost_threshold.clear();
  RW_stiction_threshold.clear();
  RW_stiction_cost.clear();
  sunpoint_axes.clear();
  sunpoint_angs.clear();
  sunpoint_useACOSs.clear();
  mtq_ax_mat = mat(3,0).zeros();
  rw_ax_mat = mat(3,0).zeros();
  // ~MTQ_axes;
  // ~RW_axes;
  // ~magic_axes;
  // ~magic_max_torq;
  // ~MTQ_max;
  // ~RW_max_torq;
  // ~RW_max_ang_mom;
}

py::tuple Satellite::py_tuple_out() const
{
  std::vector<py::array_t<double>> mtq_axes_vec;
  for (vec3 const &c: MTQ_axes) {
    mtq_axes_vec.push_back(armaVectorToNumpy(c));
  }
  std::vector<py::array_t<double>> rw_axes_vec;
  for (vec3 const &c: RW_axes) {
    rw_axes_vec.push_back(armaVectorToNumpy(c));
  }
  std::vector<py::array_t<double>> magic_axes_vec;
  for (vec3 const &c: magic_axes) {
    magic_axes_vec.push_back(armaVectorToNumpy(c));
  }
  std::vector<py::array_t<double>> sp_axes_vec;
  for (vec3 const &c: sunpoint_axes) {
    sp_axes_vec.push_back(armaVectorToNumpy(c));
  }
  return py::make_tuple(armaMatrixToNumpy(Jcom),
                        plan_for_gg,
                        py::make_tuple(plan_for_prop,armaVectorToNumpy(prop_torq)),
                        py::make_tuple(plan_for_srp,armaMatrixToNumpy(srp_coeff)),
                        py::make_tuple(plan_for_aero,armaMatrixToNumpy(drag_coeff)),
                        py::make_tuple(plan_for_resdipole,armaVectorToNumpy(res_dipole)),
                        py::make_tuple(plan_for_gendist,armaVectorToNumpy(gen_dist_torq)),
                        py::make_tuple(number_MTQ,mtq_axes_vec, MTQ_max,MTQ_cost),
                        py::make_tuple(number_RW,rw_axes_vec, RW_max_torq,RW_max_ang_mom,RW_cost,RW_AM_cost,RW_AM_cost_threshold,RW_stiction_cost,RW_stiction_threshold,RW_J),
                        py::make_tuple(number_magic,magic_axes_vec, magic_max_torq,magic_cost),
                        py::make_tuple(useAVconstraint,AVmax),
                        py::make_tuple(number_sunpoints,sp_axes_vec,sunpoint_angs,sunpoint_useACOSs)
                        );
                      }

PYBIND11_MODULE(pysat, m) {
    py::class_<Satellite>(m, "Satellite")
        .def(py::init<>())
        //.def(py::init<ALL_SETTINGS_PY_FORM>())
        .def("change_Jcom", &Satellite::change_Jcom_py)
        .def("readJcom", &Satellite::readJcom)
        .def("read_magrw_torq_mult", &Satellite::read_magrw_torq_mult)
        .def("add_gg_torq", &Satellite::add_gg_torq)
        .def("remove_gg_torq", &Satellite::remove_gg_torq)
        .def("add_prop_torq", &Satellite::add_prop_torq_py)
        .def("remove_prop_torq", &Satellite::remove_prop_torq)
        .def("add_srp_torq", &Satellite::add_srp_torq_py)
        .def("remove_srp_torq", &Satellite::remove_srp_torq)
        .def("add_aero_torq", &Satellite::add_aero_torq_py)
        .def("remove_aero_torq", &Satellite::remove_aero_torq)
        .def("add_resdipole_torq", &Satellite::add_resdipole_torq_py)
        .def("remove_resdipole_torq", &Satellite::remove_resdipole_torq)
        .def("add_gendist_torq", &Satellite::add_gendist_torq_py)
        .def("remove_gendist_torq", &Satellite::remove_gendist_torq)
        .def("add_MTQ", &Satellite::add_MTQ_py)
        .def("clear_MTQs", &Satellite::clear_MTQs)
        .def("add_magic", &Satellite::add_magic_py)
        .def("clear_magics", &Satellite::clear_magics)
        .def("add_RW", &Satellite::add_RW_py)
        .def("clear_RWs", &Satellite::clear_RWs)
        .def("auto_scale_control_costs", &Satellite::auto_scale_control_costs, py::arg("base_weight") = 1.0)
        .def("set_AV_constraint", &Satellite::set_AV_constraint)
        .def("clear_AV_constraint", &Satellite::clear_AV_constraint)
        .def("add_sunpoint_constraint", &Satellite::add_sunpoint_constraint_py)
        .def("clear_sunpoint_constraints", &Satellite::clear_sunpoint_constraints)
        .def("set_srp_surfaces", &Satellite::set_srp_surfaces_py)
        .def("clear_srp_surfaces", &Satellite::clear_srp_surfaces)
        .def("set_drag_surfaces", &Satellite::set_drag_surfaces_py)
        .def("clear_drag_surfaces", &Satellite::clear_drag_surfaces)
        .def("py_tuple_out",&Satellite::py_tuple_out)
        .def(py::pickle(
          [](const Satellite &p) { // __getstate__
              /* Return a tuple that fully encodes the state of the object */
            return p.py_tuple_out();
          },
          [](py::tuple t) { // __setstate__
              /* Create a new C++ instance */
              Satellite p = Satellite();
              p.change_Jcom_py(t[0].cast<py::array_t<double>>());
              int use_gg = t[1].cast<int>();
              if(use_gg == 1){p.add_gg_torq();}
              py::tuple prop_array = t[2].cast<py::tuple>();
              int use_prop = prop_array[0].cast<int>();
              if(use_prop == 1){
                  py::array_t<double> pt = prop_array[1].cast<py::array_t<double>>();
                  p.add_prop_torq_py(pt);
              }
              py::tuple srp_array = t[3].cast<py::tuple>();
              int use_srp = srp_array[0].cast<int>();
              if(use_srp == 1){
                  py::array_t<double> sc = srp_array[1].cast<py::array_t<double>>();
                  mat srp_coeff_in = numpyToArmaMatrix(sc);
                  p.add_srp_torq(srp_coeff_in,srp_coeff_in.n_cols);
                }
              py::tuple drag_array = t[4].cast<py::tuple>();
              int use_drag = drag_array[0].cast<int>();
              if(use_drag == 1){
                  py::array_t<double> dc = drag_array[1].cast<py::array_t<double>>();
                  mat drag_coeff_in = numpyToArmaMatrix(dc);
                  p.add_aero_torq(drag_coeff_in,drag_coeff_in.n_cols);
                }
              py::tuple rd_array = t[5].cast<py::tuple>();
              int use_rd = rd_array[0].cast<int>();
              if(use_rd == 1){
                  py::array_t<double> rd = rd_array[1].cast<py::array_t<double>>();
                  p.add_resdipole_torq_py(rd);
              }
              py::tuple gd_array = t[6].cast<py::tuple>();
              int use_gd = gd_array[0].cast<int>();
              if(use_gd == 1){
                  py::array_t<double> gd = gd_array[1].cast<py::array_t<double>>();
                  p.add_gendist_torq_py(gd);
              }
              py::tuple mtq_array = t[7].cast<py::tuple>();
              int num_mtq = mtq_array[0].cast<int>();
              if(num_mtq>0){
                std::vector<py::array_t<double>> mtq_axes = mtq_array[1].cast<std::vector<py::array_t<double>>>();
                std::vector<double> mtq_max = mtq_array[2].cast<std::vector<double>>();
                std::vector<double> mtq_cost = mtq_array[3].cast<std::vector<double>>();
                for(int k=0;k<num_mtq;k++){
                  p.add_MTQ_py(mtq_axes.at(k),mtq_max.at(k),mtq_cost.at(k));
                }
              }
              py::tuple rw_array = t[8].cast<py::tuple>();
              int num_rw = rw_array[0].cast<int>();
              if(num_rw>0){
                std::vector<py::array_t<double>> rw_axes = rw_array[1].cast<std::vector<py::array_t<double>>>();
                std::vector<double> rw_maxtorq = rw_array[2].cast<std::vector<double>>();
                std::vector<double> rw_maxAM = rw_array[3].cast<std::vector<double>>();
                std::vector<double> rw_cost = rw_array[4].cast<std::vector<double>>();
                std::vector<double> rw_AMcost = rw_array[5].cast<std::vector<double>>();
                std::vector<double> rw_AMcostThold = rw_array[6].cast<std::vector<double>>();
                std::vector<double> rw_Scost = rw_array[7].cast<std::vector<double>>();
                std::vector<double> rw_ScostThold = rw_array[8].cast<std::vector<double>>();
                std::vector<double> rw_j = rw_array[9].cast<std::vector<double>>();
                for(int k=0;k<num_rw;k++){
                  p.add_RW_py(rw_axes.at(k),rw_j.at(k),rw_maxtorq.at(k),rw_maxAM.at(k),rw_cost.at(k),rw_AMcost.at(k),rw_AMcostThold.at(k),rw_Scost.at(k),rw_ScostThold.at(k));
                }
              }
              py::tuple mgc_array = t[9].cast<py::tuple>();
              int num_mgc = mgc_array[0].cast<int>();
              if(num_mgc>0){
                std::vector<py::array_t<double>> mgc_axes = mgc_array[1].cast<std::vector<py::array_t<double>>>();
                std::vector<double> mgc_maxtorq = mgc_array[2].cast<std::vector<double>>();
                std::vector<double> mgc_cost = mgc_array[3].cast<std::vector<double>>();
                for(int k=0;k<num_mgc;k++){
                  p.add_magic_py(mgc_axes.at(k),mgc_maxtorq.at(k),mgc_cost.at(k));
                }
              }
              py::tuple av_array = t[10].cast<py::tuple>();
              bool use_av_constraint = av_array[0].cast<bool>();
              if(use_av_constraint){
                double avlim = av_array[1].cast<double>();
                p.set_AV_constraint(avlim);
              }
              py::tuple sp_array = t[11].cast<py::tuple>();
              int num_sp = sp_array[0].cast<int>();
              if(num_sp>0){
                std::vector<py::array_t<double>> sp_axes = sp_array[1].cast<std::vector<py::array_t<double>>>();
                std::vector<double> sp_angs = sp_array[2].cast<std::vector<double>>();
                std::vector<bool> sp_acoss = sp_array[3].cast<std::vector<bool>>();
                for(int k=0;k<num_sp;k++){
                  p.add_sunpoint_constraint_py(sp_axes.at(k),sp_angs.at(k),sp_acoss.at(k));
                }
              }

              return p;
        }));
}

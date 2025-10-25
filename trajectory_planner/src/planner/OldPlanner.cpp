// #include "PlannerUtil.hpp"
#include "OldPlanner.hpp"
// #include "PlannerUtil.hpp"
// #include "PlannerPython.hpp"
// #include "PlannerUtilPy.hpp"
#include "../ArmaNumpy.hpp"
// #include <stdexcept>
// #include <typeinfo>
// #include <pybind11/numpy.h>
// #define earth_mu 3.986e14
#define EIGS_NORM "inf"
#define EIGS_MULT 1.0 //(1.0/3.0)
#define EIGS_POW  0.0
#define RAND_MAX_INIT 1000.0



namespace py = pybind11;
using namespace arma;
using namespace std;

OldPlanner::OldPlanner(){
}


OldPlanner::OldPlanner(Satellite sat_in,ALL_SETTINGS_FORM allSettings){
  sat = sat_in;
  OldPlanner::updateParameters_notsat(get<0>(allSettings),get<1>(allSettings),get<2>(allSettings),get<3>(allSettings),get<4>(allSettings),get<5>(allSettings),get<6>(allSettings));

}
 ALL_SETTINGS_FORM OldPlanner::readParameters() {
  return std::make_tuple( systemSettings,  alilqrSettings,  alilqrSettings2,  initialTrajSettings,  costSettings, costSettings2, costSettings_tvlqr_full);
}
void OldPlanner::setVerbosity(bool verbosity) {
  //REMOVE BEFORE FLIGHT
  verbose = verbosity;
}
void OldPlanner::updateParameters_notsat(SYSTEM_SETTINGS_FORM systemSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings2_tmp, INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp) {
    verbose = false;

    costSettings = costSettings_tmp;
    costSettings2 = costSettings2_tmp;
    // costSettings_tvlqr = costSettings_tvlqr_tmp;
    costSettings_tvlqr_full = costSettings_tvlqr_tmp;

    initialTrajSettings = initialTrajSettings_tmp;
    bdotgain = get<0>(initialTrajSettings); // gain used in generation of trajectories for bdot
    HLangleLimit = get<1>(initialTrajSettings); //dividing line between using high settings or low settings in trajectory generation. in radians!
    highSettings = get<2>(initialTrajSettings);
    lowSettings = get<3>(initialTrajSettings);

    gyrogainH = get<0>(highSettings); //unused, to offset wxJw term in dynamics
    dampgainH = get<1>(highSettings); //used in initial trajectory generation to damp J*w
    velgainH = get<2>(highSettings); //used in initial trajectory generation to  counter J*(w-w_goal)
    quatgainH = get<3>(highSettings); //used in initial trajectory generation to  counter pointing error
    randvalH = get<4>(highSettings); //level of random noise to add to initial trajectory generation
    umaxmultH = get<5>(highSettings); //how much to reduce/increase maximum control levels for trajectory generation.

    gyrogainL = get<0>(lowSettings); //see above but for low angle settings
    dampgainL = get<1>(lowSettings);
    velgainL = get<2>(lowSettings);
    quatgainL = get<3>(lowSettings);
    randvalL = get<4>(lowSettings);
    umaxmultL = get<5>(lowSettings);

    angle_weight = get<0>(costSettings); //cost of angle error
    angvel_weight = get<1>(costSettings); //cost of av error
    u_weight = get<2>(costSettings); //cost of actuation.
    u_with_mag_weight = get<3>(costSettings); //currently unused. originally meant to further weight any alignment of u with magnetic field.
    av_with_mag_weight = get<4>(costSettings); //currently unused. orignially meant to weight the alignment of angvel with the magnetic field
    ang_av_weight = get<5>(costSettings); //cost of orientation error that aligns with av error.
    angle_weight_N = get<6>(costSettings); // cost of angle error in final timestep.
    angvel_weight_N = get<7>(costSettings);// cost of av error in final timestep.
    av_with_mag_weight_N = get<8>(costSettings); // cost of av aligned with B field error in final timestep. currently unused.
    ang_av_weight_N = get<9>(costSettings); // // cost of av/ang error alignment in final timestep.
    whichAngCostFunc = get<10>(costSettings); //determines from a variety of options how angle cost is calculated. 0-3 for vectort angle, 0-4 for quaternion.
    useRawControlCost = get<11>(costSettings);//if true (1), control cost is 0.5*u.T@W@u. if false (0), control cost is 0.5*(u-u_prev).T@W@(u-u_prev)

    angle_weight2 = get<0>(costSettings2);
    angvel_weight2 = get<1>(costSettings2);
    u_weight2 = get<2>(costSettings2);
    u_with_mag_weight2 = get<3>(costSettings2);
    av_with_mag_weight2 = get<4>(costSettings2);
    ang_av_weight2 = get<5>(costSettings2);
    angle_weight_N2 = get<6>(costSettings2);
    angvel_weight_N2 = get<7>(costSettings2);
    av_with_mag_weight_N2 = get<8>(costSettings2);
    ang_av_weight_N2 = get<9>(costSettings2);
    whichAngCostFunc2 = get<10>(costSettings2);
    useRawControlCost2 = get<11>(costSettings);

    angle_weight_tvlqr = get<0>(costSettings_tvlqr_tmp);
    angvel_weight_tvlqr = get<1>(costSettings_tvlqr_tmp);
    u_weight_tvlqr = get<2>(costSettings_tvlqr_tmp);
    u_with_mag_weight_tvlqr = get<3>(costSettings_tvlqr_tmp);
    av_with_mag_weight_tvlqr = get<4>(costSettings_tvlqr_tmp);
    ang_av_weight_tvlqr = get<5>(costSettings_tvlqr_tmp);
    angle_weight_N_tvlqr = get<6>(costSettings_tvlqr_tmp);
    angvel_weight_N_tvlqr = get<7>(costSettings_tvlqr_tmp);
    av_with_mag_weight_N_tvlqr = get<8>(costSettings_tvlqr_tmp);
    ang_av_weight_N_tvlqr = get<9>(costSettings_tvlqr_tmp);
    useRawControlCost_tvlqr = get<11>(costSettings);
    whichAngCostFunc_tvlqr = get<10>(costSettings_tvlqr_tmp);

    costSettings_tvlqr = make_tuple(angle_weight_tvlqr,angvel_weight_tvlqr,u_weight_tvlqr,
                    u_with_mag_weight_tvlqr,av_with_mag_weight_tvlqr,ang_av_weight_tvlqr,
                    angle_weight_N_tvlqr,angvel_weight_N_tvlqr,av_with_mag_weight_N_tvlqr,
                    ang_av_weight_N_tvlqr,whichAngCostFunc_tvlqr,useRawControlCost_tvlqr);
    tracking_LQR_formulation = get<12>(costSettings_tvlqr_tmp); //mode for LQR calculation.

    // systemSettings = systemSettings_tmp;
    dt = get<1>(systemSettings_tmp);
    dt_tvlqr = get<2>(systemSettings_tmp);
    eps = get<3>(systemSettings_tmp);
    tvlqr_len = get<4>(systemSettings_tmp);//lenght of TVLQR segemnts
    tvlqr_overlap = get<5>(systemSettings_tmp); //overlap between TVLQR segments
    systemSettings = make_tuple(sat.Jcom,dt,dt_tvlqr,eps,tvlqr_len,tvlqr_overlap);

    alilqrSettings = alilqrSettings_tmp;
    lineSearchSettings = get<0>(alilqrSettings_tmp);
    auglagSettings = get<1>(alilqrSettings_tmp);
    breakSettings = get<2>(alilqrSettings_tmp);
    regSettings = get<3>(alilqrSettings_tmp);

    maxLsIter = get<0>(lineSearchSettings); //max iterations on linesearch inside ilqr
    beta1 = get<1>(lineSearchSettings); //limits on how much improvement to expect in ilqr.
    beta2 = get<2>(lineSearchSettings);

    lagMultInit = get<0>(auglagSettings); //starting value of lagrange multipliers on constraints
    lagMultMax = get<1>(auglagSettings);//max value of lagrange multipliers on constraints
    penInit = get<2>(auglagSettings); //initial penalty value
    penMax = get<3>(auglagSettings); //maximum penalty value
    penScale = get<4>(auglagSettings); //scaling rate of penalty

    maxOuterIter = get<0>(breakSettings);
    maxIlqrIter = get<1>(breakSettings);
    maxIter = get<2>(breakSettings);
    gradTol = get<3>(breakSettings);
    ilqrCostTol = get<4>(breakSettings);
    costTol = get<5>(breakSettings);
    zCountLim = get<6>(breakSettings);
    cmax = get<7>(breakSettings); //currently unused.
    maxCost = get<8>(breakSettings);
    xmax = (get<9>(breakSettings));
    // breakSettings = make_tuple(maxOuterIter,maxIlqrIter,maxIter,gradTol,ilqrCostTol,costTol,zCountLim,cmax,maxCost,xmax);

    regInit = get<0>(regSettings);
    regMin = get<1>(regSettings);
    regMax = get<2>(regSettings);
    regScale = get<3>(regSettings);
    regBump = get<4>(regSettings);
    regMinConds = get<5>(regSettings);
    regBumpRandAddRatio = get<6>(regSettings);
    useEVmagic = get<7>(regSettings);
    SPDEVreg = get<8>(regSettings);
    SPDEVregAll = get<9>(regSettings);
    rhoEVregTest = get<10>(regSettings);
    EVregTestpreabs = get<11>(regSettings);
    EVaddreg = get<12>(regSettings);
    EVregIsRho = get<13>(regSettings);
    EVrhoAdd = get<14>(regSettings);
    useDynamicsHess = get<15>(regSettings);
    useConstraintHess = get<16>(regSettings);

    // alilqrSettings = make_tuple(lineSearchSettings,auglagSettings,breakSettings,regSettings);

    alilqrSettings2 = alilqrSettings2_tmp;
    lineSearchSettings2 = get<0>(alilqrSettings2_tmp);
    auglagSettings2 = get<1>(alilqrSettings2_tmp);
    breakSettings2 = get<2>(alilqrSettings2_tmp);
    regSettings2 = get<3>(alilqrSettings2_tmp);

    maxLsIter2 = get<0>(lineSearchSettings2);
    beta12 = get<1>(lineSearchSettings2);
    beta22 = get<2>(lineSearchSettings2);

    lagMultInit2 = get<0>(auglagSettings2);
    lagMultMax2 = get<1>(auglagSettings2);
    penInit2 = get<2>(auglagSettings2);
    penMax2 = get<3>(auglagSettings2);
    penScale2 = get<4>(auglagSettings2);

    maxOuterIter2 = get<0>(breakSettings2);
    maxIlqrIter2 = get<1>(breakSettings2);
    maxIter2 = get<2>(breakSettings2);
    gradTol2 = get<3>(breakSettings2);
    ilqrCostTol2 = get<4>(breakSettings2);
    costTol2 = get<5>(breakSettings2);
    zCountLim2 = get<6>(breakSettings2);
    cmax2 = get<7>(breakSettings2);
    maxCost2 = get<8>(breakSettings2);
    xmax2 = (get<9>(breakSettings2));
    // breakSettings2 = make_tuple(maxOuterIter2,maxIlqrIter2,maxIter2,gradTol2,ilqrCostTol2,costTol2,zCountLim2,cmax2,maxCost2,xmax2);

    regInit2 = get<0>(regSettings2);
    regMin2 = get<1>(regSettings2);
    regMax2 = get<2>(regSettings2);
    regScale2 = get<3>(regSettings2);
    regBump2 = get<4>(regSettings2);
    regBumpRandAddRatio2 = get<4>(regSettings2);
    useEVmagic2 = get<7>(regSettings2);
    SPDEVreg2= get<8>(regSettings2);
    SPDEVregAll2 = get<9>(regSettings2);
    rhoEVregTest2 = get<10>(regSettings2);
    EVregTestpreabs2 = get<11>(regSettings2);
    EVaddreg2 = get<12>(regSettings2);
    EVregIsRho2 = get<13>(regSettings2);
    EVrhoAdd2 = get<14>(regSettings2);
    useDynamicsHess2 = get<15>(regSettings2);
    useConstraintHess2 = get<16>(regSettings2);
}

BEFORE_OUTPUT_FORM OldPlanner::trajOptBefore(VECTOR_INFO_FORM vecs_w_time,double dt_use, TIME_FORM time_start, TIME_FORM time_end, vec x0, int bdotOn)
{
  // cout<<x0<<"\n";
  x0 = sat.state_norm(x0);
  // cout<<"inside x0"<<x0.t()<<"\n";
  // x0.rows(3, 6) = normalise(x0.rows(3,6));
  //double dt = this->dt;//readJsonDouble(trajOptSettingsFile, "dt");
  // double dt_readin = (t(1)-t(0))*36525.0*24.0*3600.0;
  cout<<"doing vec times\n";
  // vec t = get<0>(vecs_w_time);

  VECTOR_INFO_FORM vecs = findVecTimes(vecs_w_time,dt_use/(36525.0*24.0*3600.0),time_start,time_end);
  // vec times = get<0>(vecs);

  cout<<"done with vec times\n";
  mat dt_timevec = get<0>(vecs);

  int traj_length = dt_timevec.n_elem;
  // VECTOR_INFO_FORM vecs = get<1>(time_tuple);
  COST_SETTINGS_FORM costSettings_tmp = this->costSettings;

  cout<<"read in ECIvecs\n";


//  double penInit = this->penInit;//readJsonDouble(trajOptSettingsFile, "penInit");
  //double penScale = this->penScale;//readJsonDouble(trajOptSettingsFile, "penScale");
  mat ECIvec = get<6>(vecs);
  // if(verbose) {
  //   cout<<ECIvec;
  // }
  mat satvec = get<5>(vecs);

  mat nBset = normalise(get<3>(vecs));
  //

  double mu0 = 0.0;//penInit*pow(penScale,1);

  mat U = mat(sat.control_N(),traj_length).zeros();
  U.fill(datum::nan);
  mat X = mat(sat.state_N(),traj_length).fill(datum::nan);
  mat TQ = mat(3,traj_length).fill(datum::nan);
  TRAJECTORY_FORM traj;

  cout<<"setting"<<bdotOn<<endl;
  if(bdotOn==0 || sat.number_MTQ<3 || bdotOn>3)
  {
    if(verbose)
    {
      cout<<"bdotOn is false, generating random initial trajectory!";
    }
    vec umax = join_cols(vec(sat.MTQ_max),0.1*vec(sat.RW_max_torq),0.1*vec(sat.magic_max_torq));
    U = diagmat(umax)*randn(size(U))/RAND_MAX_INIT;
    if(verbose)
    {
      cout<<U;
    }
     traj = OldPlanner::generateInitialTrajectory(dt_use,x0, U, vecs);
     assert(approx_equal(get<1>(traj),U,"abstol",1e-10));
     X = get<0>(traj);
  }
  else
  {
    std::tuple<TRAJECTORY_FORM,double> bdotout = OldPlanner::bdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0);
    if(verbose)
    {
      cout<<"bdot attempted\n";
    }
    traj = std::get<0>(bdotout);
    if (bdotOn == 2)
    {
      std::tuple<TRAJECTORY_FORM,double> sbdotout = OldPlanner::smartbdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0,false);
      cout<<"smart bdot complete\n";
      traj = std::get<0>(sbdotout);
      mat u0 = get<1>(traj);
      cout<<size(get<0>(traj))<<" "<<size(get<1>(traj))<<"\n";
      TRAJECTORY_FORM traj2 = OldPlanner::generateInitialTrajectory(dt_use,x0, u0,vecs);//+diagmat(mean(abs(u0),1)*randval)*(randu(size(u0))-0.5), vecs);
      cout<<size(get<0>(traj2))<<" "<<size(get<1>(traj2))<<"\n";
      cout<<"smartbdot traj generated\n";
    }
    else if (bdotOn == 3)
    {
      std::tuple<TRAJECTORY_FORM,double> sbdotout = OldPlanner::smartbdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0,false);
      traj = std::get<0>(sbdotout);
      mat u0 = get<1>(traj);
      SMARTBDOT_SETTINGS_FORM sbSettings = this->highSettings;
      mat ECIvec = get<6>(vecs);
      mat satvec = get<5>(vecs);

      vec ek = ECIvec.col(0);
      vec3 ak = normalise(satvec.col(0));
      double ang0;

      if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
        ek = ek.tail(3);
        ang0 = acos(norm_dot(ak,rotMat(x0.rows(3,6)).t()*ek));
      }else{
        ang0  = acos(2.0*pow(norm_dot(x0.rows(3,6),ek),2.0)-1.0);
      }
      if (ang0 >= HLangleLimit){
        sbSettings = this->lowSettings;
      }
      double randval = get<4>(sbSettings);
      traj = OldPlanner::generateInitialTrajectory(dt_use,x0, u0+diagmat(max(abs(u0),1)*randval)*(2*randu(size(u0))-0.5), vecs);
    }
    U.cols(0,traj_length-1) = get<1>(traj);
    X = get<0>(traj);
    // cout<<"xend"<<X.col(0).t()<<endl;
    TQ = get<3>(traj);
    cout<<x0<<dt_use<<traj_length<<mu0<<X.has_nan()<<U.has_nan()<<"\n";
    if(X.has_nan() || U.has_nan()){
        cout<<X<<"\n";
        cout<<U<<"\n";
      }
  }
  traj = make_tuple(X,U,dt_timevec,TQ);
  if(verbose){cout<<"initial traj done\n";}


  // cout<<" traj leng before "<<X.n_cols<<"\n";
  // cout<<" traj leng before "<<U.n_cols + 1<<"\n";
  return make_tuple(traj, vecs, costSettings_tmp);
}
AFTER_OUTPUT_FORM OldPlanner::trajOptAfter(VECTOR_INFO_FORM vecs_w_time,double dt_prev, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_FORM alilqrOut)
{
  OPT_FORM opt = get<0>(alilqrOut);
  double muOut = std::get<1>(alilqrOut);
  double gradOut = std::get<2>(alilqrOut);
  double gradOut2 = gradOut;

  mat Xset = std::get<0>(opt);
  mat Uset = std::get<1>(opt);
  mat TQset = std::get<2>(opt);
  mat Kset = std::get<3>(opt);
  mat lambdaSet = std::get<4>(opt);
  vec time_vec = std::get<5>(opt);


  // int traj_length = floor((time_end-time_start)*36525.0*24.0*3600.0/dt_tvlqr);
  // cout<<traj_length<<"\n";
  // cout<<(time_end-0.22)*36525.0*24.0*3600.0<<" "<<(time_start-0.22)*36525.0*24.0*3600.0<<"\n";
  // cout<<dt_tvlqr<<" "<<(time_end-time_start)*36525.0*24.0*3600.0/dt_tvlqr<<"\n";
  VECTOR_INFO_FORM vecs_tvlqr = findVecTimes(vecs_w_time,dt_tvlqr/(36525.0*24.0*3600.0),time_start,time_end);


  vec tvlqr_times = get<0>(vecs_tvlqr);
  int traj_length =tvlqr_times.n_elem;
  cout<<" refs "<<traj_length<<"\n";
  cout<<"tvlqr times found\n";
  // VECTOR_INFO_FORM vecs_tvlqr = get<1>(time_tuple_tvlqr);
  mat Rset_tvlqr = get<1>(vecs_tvlqr);
  //COST_SETTINGS_FORM costSettings2 = this->costSettings2;
  //ALILQR_SETTINGS_FORM alilqrSettings2 = this->alilqrSettings2;

  OPT_FORM opt2;
  TRAJECTORY_FORM trajLong = make_tuple(Xset,Uset,time_vec.head(Xset.n_cols),TQset);
  if(verbose) {
    cout<<"completed ALILQR successfully\n";
    cout<<dt_prev<<" "<<dt_tvlqr<<endl;
  }
  if(dt_prev/dt_tvlqr > 1){
    double colMissing = max(0,int(Rset_tvlqr.n_cols) - 1 - int((int(Uset.n_cols)-2)*dt_prev/dt_tvlqr));

    if(verbose) {
      cout<<"colMiss: "<<colMissing<<"\n";
    }
    mat UsetLong = join_rows(repelem(Uset.cols(0,Uset.n_cols-3),1,int(dt_prev/dt_tvlqr)),repelem(Uset.cols(Uset.n_cols-2,Uset.n_cols-2),1,colMissing),Uset.tail_cols(1));

    trajLong = OldPlanner::generateInitialTrajectory(dt_tvlqr,Xset.col(0), UsetLong, vecs_tvlqr);

    // mat xtmp = get<0>(trajLong);
    // cout<<" traj leng mid "<<xtmp.n_cols<<"\n";
    // cout<<" refs "<<int(Rset_tvlqr.n_cols)<<" "<<int(Uset.n_cols)*dt/dt_tvlqr<<" "<<int(Uset.n_cols)<<" "<<Uset.n_cols<<"\n";
    // cout<<" traj leng mid "<<UsetLong.n_cols<<"\n";
    // TRAJECTORY_FORM traj_tvlqr = trajLong;
    // TRAJECTORY_FORM opt2 = trajLong;
    ALILQR_OUTPUT_FORM alilqrOut2 = OldPlanner::alilqr(dt_tvlqr,trajLong, vecs_tvlqr, costSettings2,alilqrSettings2,false);

     opt2 = get<0>(alilqrOut2);
    // double muOut2 = std::get<1>(alilqrOut2);
     gradOut2 = std::get<2>(alilqrOut2);
    // cout<<"completed full length ALILQR successfully\n";
    if(verbose) {
      cout<<"completed full length ALILQR successfully\n";
    }
  }else{
     opt2 = opt;
     tvlqr_times = time_vec;
  }

  mat K_lqr;
  mat S_lqr;
  if(tracking_LQR_formulation==0){
    K_lqr = mat(sat.control_N()*sat.reduced_state_N(),0,fill::zeros);
  }else if(tracking_LQR_formulation==2){
    K_lqr = mat(sat.control_N()*(sat.reduced_state_N()+3),0,fill::zeros);
  }else{
    K_lqr = mat(sat.control_N()*sat.reduced_state_N(),0,fill::zeros);
  }
  // K_lqr.zeros();
  if(tracking_LQR_formulation==0){
  }else if(tracking_LQR_formulation==2){
    S_lqr = mat((sat.reduced_state_N()+3)*(sat.reduced_state_N()+3),0,fill::zeros);
  }else{
    S_lqr = mat(sat.reduced_state_N()*sat.reduced_state_N(),0,fill::zeros);
  }
  // S_lqr.zeros();
  double tvlqr_overlap_tmp = floor(tvlqr_overlap/dt_tvlqr)*dt_tvlqr;
  double tvlqr_len_tmp = floor(tvlqr_len/dt_tvlqr)*dt_tvlqr;

  TIME_FORM time_start_tmp = time_start;
  TIME_FORM time_end_tmp = time_start + (tvlqr_overlap_tmp-0*dt_tvlqr)/(36525.0*24.0*3600.0);
  // cout<<(time_end_tmp-time_start)*36525.0*24.0*3600.0<<" "<<(time_start_tmp-time_start)*36525.0*24.0*3600.0<<"\n";
  double col0 = 0 ;
  double col1 = col0 + (tvlqr_overlap_tmp/dt_tvlqr - 0*1);
  double col1u = 0;
  cout<<"time to find K\n";

  mat K2 = get<3>(opt2);
  mat U_lqr = get<1>(opt2);
  mat X_lqr = get<0>(opt2);
  // cout<<" traj leng "<<X_lqr.n_cols<<"\n";
  mat TQ_lqr = get<2>(opt2);
  vec dt_lqr = get<5>(opt2);

  mat K2_tmp;
  mat U_lqr_tmp;
  mat X_lqr_tmp;
  mat TQ_lqr_tmp;
  vec dt_lqr_tmp;

  do{
      // cout<<"01 "<<time_start_tmp<<" "<<time_end_tmp<<"\n";
      time_start_tmp = time_end_tmp - (tvlqr_overlap_tmp-0*dt_tvlqr)/(36525.0*24.0*3600.0);
      time_end_tmp = min(time_start_tmp + (tvlqr_len_tmp+0*dt_tvlqr)/(36525.0*24.0*3600.0),time_end-0*dt_tvlqr/(36525.0*24.0*3600.0));
      //double dt_tvlqr = this->dt_tvlqr;//readJsonDouble(trajOptSettingsFile, "dt");
      // int traj_length2 = floor(N/dt_tvlqr);
      col0 = col1 - (tvlqr_overlap_tmp/dt_tvlqr + 0*1);
      col1 = min(double(X_lqr.n_cols)-1,col0 + tvlqr_len_tmp/dt_tvlqr + 0*1);
      col1u = min(double(U_lqr.n_cols)-1,col0 + tvlqr_len_tmp/dt_tvlqr + 0*1);

      // cout<<tvlqr_overlap_tmp<<"\n";
      // cout<<tvlqr_len_tmp<<"\n";
      // cout<<(time_end_tmp-time_start)*36525.0*24.0*3600.0<<" "<<(time_start_tmp-time_start)*36525.0*24.0*3600.0<<"\n";
      // cout<<dt_tvlqr<<" "<<(time_end_tmp-time_start_tmp)*36525.0*24.0*3600.0/dt_tvlqr<<"\n";
      // cout<<(time_start_tmp-0.22)*36525.0*24.0*3600.0<<" "<<(time_end_tmp-0.22)*36525.0*24.0*3600.0<<" "<<(time_start-0.22)*36525.0*24.0*3600.0<<" "<<(time_end-0.22)*36525.0*24.0*3600.0<<"\n";
      VECTOR_INFO_FORM vecs_tvlqr_tmp = findVecTimes(vecs_w_time,dt_tvlqr/(36525.0*24.0*3600.0),time_start_tmp,time_end_tmp+0*dt_tvlqr/(36525.0*24.0*3600.0));


      cout<<col0<<" "<<col1<<" "<<col1u<<" "<<Rset_tvlqr.n_cols<<"\n";

      dt_lqr_tmp = dt_lqr.subvec(col0,col1+0*1);
      X_lqr_tmp = X_lqr.cols(col0,col1+0*1);
      U_lqr_tmp = U_lqr.cols(col0,col1u+0*1);
      TQ_lqr_tmp = TQ_lqr.cols(col0,col1u+0*1);
      vec rd_offset = join_cols(sat.mtq_ax_mat.t()*sat.res_dipole*sat.plan_for_resdipole,vec(sat.control_N()-sat.number_MTQ).zeros());//TODO--adjsut for if there are more than 3 MTQ
      U_lqr_tmp.each_col() -= rd_offset;

      TRAJECTORY_FORM traj_tvlqr_tmp = make_tuple(X_lqr_tmp,U_lqr_tmp,dt_lqr_tmp,TQ_lqr_tmp);

      //TODO: add findK (from old planner code)
      COST_SETTINGS_FORM costSettingsFindK = this->costSettings_tvlqr;
      cube Kcube;
      cube Scube;
      std::tuple<cube, cube> KS = make_tuple(Kcube,Scube);
      if(tracking_LQR_formulation==0){
        KS = OldPlanner::findK(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp,  costSettingsFindK);
      }else if(tracking_LQR_formulation==2){
        KS = OldPlanner::findKwDist(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp,  costSettingsFindK);
      }else{
        KS = OldPlanner::findK(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp,  costSettingsFindK);
      }
      // std::tuple<cube, cube> KS = OldPlanner::findK(dt_tvlqr, traj_tvlqr, vecs_tvlqr,  costSettingsFindK);
      // std::tuple<cube, cube> KS = OldPlanner::findTrackK(dt_tvlqr, traj_tvlqr, vecs_tvlqr,  costSettingsFindK);
      cout<<"test\n";

      mat K_lqr_tmp = packageK(get<0>(KS));
      mat S_lqr_tmp = packageS(get<1>(KS));
      // cout<<"test1\n";
      // cout<<K_lqr.n_cols<<"\n";
      // cout<<K_lqr.n_cols-tvlqr_overlap_tmp<<"\n";
      // cout<<S_lqr.n_cols<<"\n";
      // cout<<S_lqr.n_cols-tvlqr_overlap_tmp-1<<"\n";
      double Klen = ((K_lqr.n_cols-tvlqr_overlap_tmp) < 0) ? 0 : K_lqr.n_cols-tvlqr_overlap_tmp;
      K_lqr = join_rows(K_lqr.head_cols(Klen),K_lqr_tmp);
      cout<<"test2\n";
      double Slen = ((S_lqr.n_cols-tvlqr_overlap_tmp-1) < 0) ? 0 : S_lqr.n_cols-tvlqr_overlap_tmp-1;
      S_lqr = join_rows(S_lqr.head_cols(Slen),S_lqr_tmp);

      cout<<size(K_lqr)<<"\n";
      cout<<size(S_lqr)<<"\n";
      // tvlqr_times = join_cols(tvlqr_times,tvlqr_times_tmp);

  }
  while((time_end_tmp<(time_end+0.0/(36525.0*24.0*3600.0))-EPSVAR)&&(time_start_tmp<(time_end-tvlqr_overlap_tmp/(36525.0*24.0*3600.0))));
  cout<<"K found\n";
  // cout<<"time vec"<<time_vec;
  // cout<<"get<0>(opt2)"<<get<0>(opt2)<<"\n";


  // OPT_TIMES_FORM main_opt_times = (addOptTimes(opt));
  OPT_FORM lqr_opt = make_tuple(get<0>(opt2),get<1>(opt2),get<2>(opt2),K_lqr,S_lqr,tvlqr_times.head(get<0>(opt2).n_cols));
  //return success
  return std::make_tuple(1, gradOut2, opt2, lqr_opt, trajLong);
}

AFTER_OUTPUT_FORM OldPlanner::trajOpt(VECTOR_INFO_FORM &vecs,int N, TIME_FORM time_start, TIME_FORM time_end, vec x0, int bdotOn)
{

  // VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  // vec x0 = numpyToArmaVector(x0Numpy);

  cout<<"in c++\n";
  cout<<x0<<"\n";
  BEFORE_OUTPUT_FORM results = OldPlanner::trajOptBefore(vecs, dt, time_start, time_end, x0, bdotOn);
  cout<<"past before\n";
  TRAJECTORY_FORM traj_init = get<0>(results);
  VECTOR_INFO_FORM vecs_dt = get<1>(results);
  COST_SETTINGS_FORM costSettings_tmp = get<2>(results);
  // mat dt_timevec = get<3>(results);
  cout<<"past before wrapup\n";
  //ALILQR_SETTINGS_FORM alilqrSettings = this->alilqrSettings;
  ALILQR_OUTPUT_FORM alilqrOut = OldPlanner::alilqr(dt,traj_init, vecs_dt, costSettings_tmp,alilqrSettings,false);
  cout<<"out of alilqr\n";

  //any disturbances?

  // OPT_FORM opt_tmp0 = get<0>(alilqrOut);
  // mat Xtmp0 = get<0>(opt_tmp0);
  // mat Utmp0 = get<1>(opt_tmp0);
  // // cout<<" traj leng after 1 "<<Xtmp0.n_cols<<"\n";
  // // cout<<" traj leng after 1 "<<Utmp0.n_cols + 1<<"\n";
  //
  // bool anyDist = bool(sat.plan_for_gg || sat.plan_for_srp || sat.plan_for_aero || sat.plan_for_prop || sat.plan_for_resdipole || sat.plan_for_gendist);
  // // assert(!anyDist);
  // cout<<"any dist?\n";
  // if(anyDist){
  //   cout<<"doing dist\n";
  //   //if any disturbances, run the basic/larger-time-step alilqr again, with the disturbances on.
  //   OPT_FORM opt = get<0>(alilqrOut);
  //     cout<<"doing dist1\n";
  //   TRAJECTORY_FORM traj_0 = make_tuple(get<0>(opt),get<1>(opt),get<5>(opt),get<2>(opt));
  //     cout<<"doing dist2\n";
  //   alilqrOut = OldPlanner::alilqr(dt,traj_0, vecs_dt, costSettings_tmp,alilqrSettings,false);
  // }
  OPT_FORM opt_tmp = get<0>(alilqrOut);
  mat Xtmp = get<0>(opt_tmp);
  mat Utmp = get<1>(opt_tmp);
  // cout<<" traj leng after 2 "<<Xtmp.n_cols<<"\n";
  // cout<<" traj leng after 2 "<<Utmp.n_cols + 1<<"\n";
  cout<<"dist done\n";
  AFTER_OUTPUT_FORM results2 = OldPlanner::trajOptAfter(vecs, dt, time_start, time_end, alilqrOut);
  cout<<"last bit done\n";
  return results2;
}



/*This function generates bdot gains to initialize the trajectory optimizer
  Inputs:
    Bset - magnetic field over time
    x0 - initial x position
    bdotgain - bdot gain
    umax - max allowable dipole
    dt - delta t
*/
tuple<TRAJECTORY_FORM,double> OldPlanner::bdot(vec x0,double dt0, int N,VECTOR_INFO_FORM vecs,  COST_SETTINGS_FORM costSettings_tmp,double mu)
{
  mat lambdaSet = mat(sat.constraint_N(), N).zeros();
  mat muSet0 = mat(sat.constraint_N(), N).ones()*mu;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet0);
  //Initialize newX
  //int N = Bset.n_cols+1;
  mat Xset = mat(sat.state_N(), N);
  mat Uset = mat(sat.control_N(), N).zeros();
  mat TQset = mat(3, N).zeros();
  cout.precision(4);

  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec t = get<0>(vecs);
  //Xset.fill(datum::nan);

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  Xset.col(0) = xk;
  vec4 qk = xk.rows(3, 6);
  cout<<"test bdot\n";
  vec3 Bk = Bset.col(0);
  mat33 RmatT = rotMat(qk).t();
  vec uk = vec(sat.control_N()).zeros();
  vec umax = join_cols(vec(sat.MTQ_max),vec(sat.RW_max_torq),vec(sat.magic_max_torq));

  double ur = max(abs(uk/umax));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  uk.head(sat.number_MTQ) = -sat.mtq_ax_mat.t()*bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(1)-Bk)/dt0);
  ur = max(abs(uk/umax));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),0);
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {
    // cout<<k<<" "<<Bset.col(k).t()<<"\n";

    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),0);
    RmatT = rotMat(qk).t();
    // uk = -bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(k)-Bk)/dt0);
    uk.zeros();
    // cout<<xk.t()<<"\n";
    uk.head(sat.number_MTQ) = -sat.mtq_ax_mat.t()*bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(k)-Bk)/dt0);///pow(norm(Bk),2);

    ur = max(abs(uk/umax));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    if(sat.number_RW>0){
      uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = 0*-diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
    }
    ur = max(abs(uk/umax));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    // cout<<uk.t()<<"\n";
    Uset.col(k-1) = uk;
    //xprev = xk;
    Bk = Bset.col(k);
    dynout = rk4z(dt0,xk, uk, sat,dynamics_info_kn1, dynamics_info_k);
    vec dos = get<0>(dynout);
    xk = sat.state_norm(get<0>(dynout));
    // qk = normalise(xk.rows(3, 6));
    // xk.rows(3, 6) = qk;
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);
    // cout<<k<<" "<<dos.t()<<" "<<xk.t()<<" "<<uk.t()<<" "<<(sat.invJcom_noRW*cross(uk,Bset.col(k-1))).t()<<" "<<20*(sat.invJcom_noRW*cross(uk,Bset.col(k-1))).t()<<" "<<TQset.col(k-1).t()<<"\n";
    // cout<<"\n\n";

  }
  TRAJECTORY_FORM traj = make_tuple(Xset,Uset,t,TQset);
  double bcost = OldPlanner::cost2Func(traj, vecs, auglag_vals, &costSettings_tmp);
  return make_tuple(traj,bcost);
}
/*This function generates a control trajectory to initialize the trajectory optimizer that accounts for goals
  Inputs:
    Bset - magnetic field over time
    ECIvec - desired pointing of satvec over time
    satvec - pointing vec over time
    x0 - initial x position
    dampgain - bdot gain (velocity damping), now scaled by norm(B)^2
    velgain - gain to control attempt to match desired angular velocity
    quatgain - gain to control attempt to match vector pointing
    umax - max allowable dipole
    dt0 - delta t
*/
tuple<TRAJECTORY_FORM,double> OldPlanner::smartbdot(vec x0,double dt0,int N,VECTOR_INFO_FORM vecs,COST_SETTINGS_FORM costSettings_tmp,double mu,bool invert)
{


  //double HLangleLimit = this->HLangleLimit;
  SMARTBDOT_SETTINGS_FORM sbSettings = highSettings;
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  vec ek = ECIvec.col(0);



  mat lambdaSet = mat(sat.constraint_N(), N).zeros();
  mat muSet0 = mat(sat.constraint_N(), N).ones()*mu;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet0);
  //int N = Bset.n_cols+1;
  mat Xset = mat(sat.state_N(), N);
  mat Uset = mat(sat.control_N(), N).zeros();
  mat TQset = mat(3, N).zeros();

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  Xset.col(0) = xk;
  vec4 qk = normalise(xk.rows(3, 6));

  //Initialize stuff for inside loop
  //vec xprev = vec(x0);
  //vec3 uk = vec(3).zeros();
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec t = get<0>(vecs);
  vec3 Bk = Bset.col(0);
  double nB2 = dot(Bk,Bk);
  vec3 satvk = normalise(satvec.col(0));
  double ang0;

  if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
    ek = ek.tail(3);
     ang0 = acos(norm_dot(satvk,rotMat(qk).t()*ek));
  }else{
     ang0  = acos(2.0*pow(norm_dot(qk,ek),2.0)-1.0);
  }
  if (ang0 >= HLangleLimit){
    sbSettings = lowSettings;
  }
  double umaxmult = get<5>(sbSettings);

  vec umax = join_cols(vec(sat.MTQ_max),0.01*vec(sat.RW_max_torq),0.01*vec(sat.magic_max_torq));
  vec umax2 = umaxmult*umax;//0.75*umax;


  // ////cout<<"test";
  //
  // double invsign = 1.0;
  // if (invert){invsign = -1.0;};

  // mat33 wt = mat33().eye();//sat.Jcom;

  mat33 RmatT = rotMat(qk).t();
  vec3 Bbody = RmatT*Bk;
  //uk = -bdotgain*(-cross(newX(1:3,1),rotT(newX(end-3:end,1))*Bset(:,1)) + rotT(newX(end-3:end,1))*(Bset(:,2)-Bset(:,1))/dt0);
  vec ECIvk = ECIvec.col(0);
  vec ECIvkp1 = ECIvec.col(1);
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),1);

  vec3 dist_torq = sat.dist_torque(xk,dynamics_info_kn1);
  vec uk = OldPlanner::smartbdot_rawmtq_finder(dt0,xk,nB2, ECIvk, ECIvkp1, satvk, Bbody,sbSettings,dist_torq);
  double ur = max(abs(uk/umax2));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  if(sat.number_RW>0){
    uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = -diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
  }
  ur = max(abs(uk/umax2));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  //uk = uk/clamp(max(abs(uk/umax)),1,datum::inf);
  Uset.col(0) = uk;

  //Loop from k = 1 to N-2 and fill in Xset, using rk4

  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {
    // cout<<k<<" "<<Bset.col(k).t()<<"\n";
    Bk = Bset.col(k);
    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bk,Rset.col(k),pset(k),Vset.col(k),Sset.col(k),1);
    Uset.col(k) = uk;
    //uk = Uset.col(k-1);
    //xprev = xk;
    dynout = rk4z(dt0,xk, uk,sat,dynamics_info_kn1,dynamics_info_k);
    xk = sat.state_norm(get<0>(dynout));
    // qk = normalise(xk.rows(3, 6));
    // xk.rows(3, 6) = qk;
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);

    nB2 = dot(Bk,Bk);
    //uk = -bdotgain*(-cross(newX(1:3,1),rotT(newX(end-3:end,1))*Bset(:,1)) + rotT(newX(end-3:end,1))*(Bset(:,2)-Bset(:,1))/dt0);
    ECIvk = ECIvec.col(k);
    if(k<N-1){ECIvkp1 = ECIvec.col(k+1);}
    satvk = satvec.col(k);
    RmatT = rotMat(qk).t();
    Bbody = RmatT*Bk;
    dist_torq = sat.dist_torque(xk,dynamics_info_k);
    uk = OldPlanner::smartbdot_rawmtq_finder(dt0,xk,nB2, ECIvk, ECIvkp1, satvk, Bbody,sbSettings,dist_torq);

    ur = max(abs(uk/umax2));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    if(sat.number_RW>0){
      uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = -diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
    }
    ur = max(abs(uk/umax2));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    // cout<<uk.t()<<"\n";
    // cout<<norm_dot(sat.Jcom*xk.head(3),-cross(Bbody,uk))<<"\n";

  }
  TRAJECTORY_FORM traj = make_tuple(Xset,Uset,t,TQset);

  double bcost = OldPlanner::cost2Func(traj, vecs,auglag_vals,  &costSettings_tmp);
  return make_tuple(traj,bcost);
}

vec OldPlanner::smartbdot_rawmtq_finder(double dt0, vec xk, double nB2,vec ECIvk, vec ECIvkp1, vec3 satvk, vec3 Bbody,SMARTBDOT_SETTINGS_FORM sbSettings,vec3 dist_torq){

  double gyrogain = get<0>(sbSettings);
  double dampgain = get<1>(sbSettings);
  double velgain = get<2>(sbSettings);
  double quatgain = get<3>(sbSettings);
  double umaxmult = get<5>(sbSettings);
  bool ek_is_3 = ((ECIvk.n_elem==3)||((ECIvk.n_elem==4)&&(isnan(ECIvk(0)))));
  bool ekp1_is_3 = ((ECIvkp1.n_elem==3)||((ECIvkp1.n_elem==4)&&(isnan(ECIvkp1(0)))));

  xk = sat.state_norm(xk);
  vec4 qk = normalise(xk.rows(3, 6));
  mat33 RmatT = rotMat(qk).t();
  vec3 wk = xk.head(3);
  vec uk = vec(sat.control_N()).zeros();


  if(ek_is_3&&ekp1_is_3){
    ECIvk = ECIvk.tail(3);
    ECIvkp1 = ECIvkp1.tail(3);
    vec3 wkdes = cross(ECIvk,ECIvkp1);
    wkdes = asin(norm(wkdes))*normalise(wkdes)/dt0;
    wkdes = RmatT*wkdes;
    vec3 ECIvkBody = RmatT*ECIvk;
    vec3 qq = sat.invJcom*normalise(cross(normalise(ECIvkBody+satvk),normalise(cross(ECIvkBody,satvk))));
    qq = normalise(qq);
    vec3 x3 = normalise(Bbody);//normalise(sat.Jcom*cross(xx,x2));

      //uk = cross(Bbody,sat.Jcom*(dampgain*(wk-dot(wk,x3)*x3) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(cross(ECIvkBody,satvk))))/nB2;
    uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom*wk) + velgain*sat.Jcom*(wk-wkdes) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(sat.Jcom*cross(ECIvkBody,satvk))))/nB2;
    if(norm(cross((qq),x3))<0.02){
      vec3 x4 = normalise(cross(qq,x3));
      uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom*wk) + velgain*sat.Jcom*(wk-wkdes) + quatgain*acos(norm_dot(ECIvkBody,satvk))*x4*sign(norm_dot(x4,normalise(cross(ECIvkBody,satvk))))))/nB2;
    }
  }else{
    if(!ek_is_3 && ekp1_is_3){
      //current is a quaternion specification. make the other a quat specification nearby

      ECIvkp1 = ECIvkp1.tail(3);
      ECIvkp1 = normalise(ECIvkp1);
      ECIvkp1 = closestQuatForVecPoint(ECIvk,satvk,ECIvkp1);
    }
    if(!ekp1_is_3 && ek_is_3){
      ECIvk = ECIvk.tail(3);
      ECIvk = normalise(ECIvk);
      ECIvk = closestQuatForVecPoint(ECIvkp1,satvk,ECIvk);
    }
    vec4 dq = normquaterr(ECIvk,ECIvkp1);
    vec4 qerr = normquaterr(ECIvk,qk);
    if(as_scalar(qerr(0))!=0){
      qerr *= sign(qerr(0));
    }
    vec3 wkdes = normalise(dq.tail(3))*2.0*asin(norm(dq.tail(3)))/dt0;
    wkdes = RmatT*wkdes;

    vec3 qq = normalise(sat.Jcom*qerr.tail(3));
    qq = normalise(qq);
    vec3 bn = normalise(Bbody);//normalise(sat.Jcom*cross(xx,x2));

      //uk = cross(Bbody,sat.Jcom*(dampgain*(wk-dot(wk,x3)*x3) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(cross(ECIvkBody,satvk))))/nB2;
    uk.head(sat.number_MTQ) = (sat.mtq_ax_mat.t()*cross(Bbody,(-dist_torq + dampgain*(sat.Jcom*wk) + velgain*(sat.Jcom*(wk-wkdes)) + quatgain*2.0*acos(as_scalar(qerr(0)))*qq)))/nB2;
    cout<<cross(Bbody,(dampgain*(sat.Jcom*wk))).t()/nB2<<"\n";
    cout<<cross(Bbody,velgain*sat.Jcom*(wk-wkdes)).t()/nB2<<"\n";
    cout<<cross(Bbody,quatgain*2.0*acos(as_scalar(qerr(0)))*qq).t()/nB2<<"\n";
    cout<<uk.t()<<"\n";
    cout<<dq.t()<<"\n";
    cout<<norm_dot(wk,qerr.tail(3))<<"\n";
    // cout<<wk.t()<<"\n";
    // cout<<sat.number_MTQ<<"\n";
    // cout<<norm_dot(sat.Jcom*wk,-cross(Bbody,uk))<<"\n";
    // if(norm(cross((qq),x3))<0.02){
    //   vec3 x4 = normalise(cross(qq,bn));
    //   uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom_noRW*wk) + velgain*sat.Jcom_noRW*(wk-wkdes) + quatgain*2*acos(qerr(0))*x4*sign(norm_dot(x4,qq))))/nB2;
    // }
  }
  return uk;
}

/* This function finds the TVLQR gains after alilqr is run
  Inputs:
    Xset, Uset, Rset, Vset, Bset - final states, control vectors, orbital position velocity and magfield
    QN - final time Q - 6 x 6 matrix
    R - control cost - 3 x 3 matrix
    dt0 - double
    satAlignVector, vNslew - 3 x 1 vectors for alignment
    costSettings - settings to find q
  Outputs:
   Kset - 3 x 6 x N-1 cube, TVLQR gains
   Sset - 6 x 6 x N cube, intermediate values used to find gains
*/
tuple<cube, cube> OldPlanner::findK(double dt_tvlqr0, TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, COST_SETTINGS_FORM costSettings_tmp)
{
  //Initialize Sset and Kset
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat sunset = get<4>(vecs);
  mat Vset = get<2>(vecs);
  vec pset = get<7>(vecs);
  int N = Xset.n_cols;
  if(verbose) {
    cout<<"N is: "<<N<<"\n";
  }
  cube Kset_lqr = cube(sat.control_N(), sat.reduced_state_N(), N-1).zeros();
  cube Sset = cube(sat.reduced_state_N(), sat.reduced_state_N(), N).zeros();

  // cout<<size(Kset_lqr)<<"\n";
  // cout<<size(Sset)<<"\n";
  //Initialize various states & properties at time k
  int k = N-1;
  // int tk = dt_tvlqr0*(k-1)+1;
  //vec rk = Rset.col(tk);
  vec xk = Xset.col(k);
  // cout<<"col \n";
  vec3 bk = Bset.col(k);
  // cout<<"col \n";
  vec3 sk = satvec.col(k);
  // cout<<"col \n";
  vec ek = ECIvec.col(k);
  // cout<<"col \n";
  vec uk = vec(sat.control_N()).zeros();
  // cout<<"col2 \n";
  vec ukp = vec(sat.control_N()).zeros();
  // cout<<"col \n";
  vec4 qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
  // cout<<"col11 \n";
  //Find lkxx = LQR Q because it's the state cost matrix
  cost_jacs costJac = sat.costJacobians(k, N, xk,uk,ukp, sk,ek,bk, &costSettings_tmp);
  // cout<<"col1 \n";
  mat lkxx = costJac.lxx;
  // cout<<"col \n";
  mat lkuu = costJac.luu;
  // cout<<"col \n";
  // cout<<"0\n";
  mat Sk = lkxx;//get<0>(weights);// mat66().zeros();
  mat Kk = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  // cout<<"1\n";
  Sset.slice(k) = Sk;//mat66().zeros();
  mat A = mat(sat.state_N(),sat.state_N()).zeros();
  mat B = mat(sat.state_N(),sat.control_N()).zeros();
  // cout<<"2\n";
  mat Aqk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  mat Bqk = mat(sat.reduced_state_N(),sat.control_N()).zeros();

  mat C = mat(sat.state_N(),3).zeros();
  //Find Gk and initialize Gkp1 and Skp1
  mat Gk = sat.findGMat(qk);
  // cout<<"3\n";
  mat Gkp1 = Gk;
  mat Skp1 = Sk;
  //vec3 prop_torq = this->prop_torq;

  //Loop backwards
  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
  // cout<<"col long\n";
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kp1;
  vec eigvals;
  mat eigvecs;

  for(int k=N-2; k>=0; k--)
  {
    dynamics_info_kp1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
    //Update states and stuff at time k
    Gkp1 = Gk;
    // tk = dt_tvlqr0*(k-1)+1;
    //rk = Rset.col(tk);
    xk = Xset.col(k);
    uk = Uset.col(k);
    ukp = ukp.zeros();
    if(k>0){ukp = Uset.col(k-1);}
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    // cout<<"4\n";
    //vk = Vset.col(tk);
    qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
    bk = Bset.col(k);
    Skp1 = Sk;
    //Get lkxx = LQR Q because it's the state cost matrix
    costJac = sat.costJacobians(k, N, xk, uk,ukp,sk,ek, bk,&costSettings_tmp);
    lkxx = costJac.lxx;
    lkuu = costJac.luu;
    // cout<<"5\n";
    //Get Gk
    Gk = sat.findGMat(qk);
    //Get A, B
    tuple<mat, mat,mat> AB = rk4zJacobians(dt_tvlqr0,xk, uk, sat,dynamics_info_k,dynamics_info_kp1);
    A = get<0>(AB);//px4MatToArma(&A_px4);
    B = get<1>(AB);//px4MatToArma(&B_px4);
    C = get<2>(AB);
    //Get Aqk and Bqk
    Aqk = Gkp1*A*trans(Gk);
    Bqk = Gkp1*B;
    // cout<<"6\n";
    // cout<<k<<"\n";
    // cout<<B<<"\n";
    // cout<<Vset.col(k).t()<<"\n";
    // cout<<Bset.col(k).t()<<"\n";
    // cout<<Rset.col(k).t()<<"\n";
    // cout<<(lkuu + trans(Bqk)*Skp1*Bqk)<<"\n";
    // cout<< (trans(Bqk)*Skp1*Aqk)<<"\n";
    Kk = solve((lkuu + trans(Bqk)*Skp1*Bqk), (trans(Bqk)*Skp1*Aqk),solve_opts::likely_sympd+solve_opts::no_approx);//+solve_opts::refine);//inv(R + trans(Bqk)*Skp1*Bqk)*(trans(Bqk)*Skp1*Aqk);//

    Kset_lqr.slice(k) = Kk;
    // Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk = lkxx + trans(Kk)*lkuu*Kk + solve((Aqk-Bqk*Kk), Skp1*(Aqk-Bqk*Kk));
    // Sk = lkxx + trans(Kk)*lkuu*Kk + trans(Aqk-Bqk*Kk)*Skp1*(Aqk-Bqk*Kk);
    Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;

    Sk = 0.5*(Sk+trans(Sk));
    // Sk = 0.5*(Sk+lkxx);
    Sset.slice(k) = Sk;
    // cout<<k<<" done of "<<N<<"\n";
  }
  cout<<size(Kset_lqr)<<"\n";
  cout<<size(Sset)<<"\n";
  return make_tuple(Kset_lqr, Sset);
}
tuple<cube, cube> OldPlanner::findKwDist(double dt_tvlqr0, TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, COST_SETTINGS_FORM costSettings_tmp)
{
  //Initialize Sset and Kset
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  vec tset = get<0>(vecs);
  mat sunset = get<4>(vecs);
  mat Vset = get<2>(vecs);
  vec pset = get<7>(vecs);
  int N = Xset.n_cols;
  if(verbose) {
    cout<<"N is: "<<N<<"\n";
  }

  cube Kset_lqr = cube(sat.control_N(), sat.reduced_state_N()+3, N-1).zeros();
  cube Sset = cube(sat.reduced_state_N()+3, sat.reduced_state_N()+3, N).zeros();
  cout<<"test\n";
  // cout<<size(Kset_lqr)<<"\n";
  // cout<<size(Sset)<<"\n";

  //Initialize various states & properties at time k
  int k = N-1;
  // int tk = dt_tvlqr0*(k-1)+1;
  //vec rk = Rset.col(tk);
  vec xk = Xset.col(k);
  cout<<"col\n";
  vec3 ek = ECIvec.col(k);
  cout<<"col\n";
  vec3 sk = satvec.col(k);
  cout<<"col\n";
  vec3 bk = Bset.col(k);
  cout<<"col\n";
  vec uk = vec(sat.control_N()).zeros();
  vec ukp = vec(sat.control_N()).zeros();
  //vec vk = Vset.col(tk);
  vec4 qk = xk.rows(sat.quat0index(),sat.quat0index()+3);

  //Find lkxx = LQR Q because it's the state cost matrix
  cost_jacs costJac = sat.costJacobians(k, N, xk,uk,ukp, sk,ek,bk, &costSettings_tmp);
  mat lkxx = costJac.lxx;
  mat lkuu = costJac.luu;
  mat Sk = mat(sat.reduced_state_N()+3,sat.reduced_state_N()+3).zeros();
  Sk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) = lkxx;//get<0>(weights);// mat66().zeros();
  Sk(span(sat.reduced_state_N(),sat.reduced_state_N()+2),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) += mat33().eye();

  mat Kk = mat(sat.control_N(),sat.reduced_state_N()+3).zeros();
  Sset.slice(k) = Sk;//mat66().zeros();
  mat A = mat(sat.state_N(),sat.state_N()).zeros();
  mat B = mat(sat.state_N(),sat.control_N()).zeros();
  mat Aqk = mat(sat.reduced_state_N()+3,sat.reduced_state_N()+3).zeros();
  mat Bqk = mat(sat.reduced_state_N()+3,sat.control_N()).zeros();

  mat C = mat(sat.state_N(),3).zeros();

  //Find Gk and initialize Gkp1 and Skp1
  mat Gk = sat.findGMat(qk);
  mat Gkp1 = Gk;
  mat Skp1 = Sk;
  //vec3 prop_torq = this->prop_torq;

  //Loop backwards
  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
  cout<<"col long\n";
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kp1;
  vec eigvals;
  mat eigvecs;

  for(int k=N-2; k>=0; k--)
  {

    dynamics_info_kp1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
    //Update states and stuff at time k
    Gkp1 = Gk;
    // tk = dt_tvlqr0*(k-1)+1;
    //rk = Rset.col(tk);
    xk = Xset.col(k);
    uk = Uset.col(k);
    ukp = ukp.zeros();
    if(k>0){ukp = Uset.col(k-1);}
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    //vk = Vset.col(tk);
    qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
    bk = Bset.col(k);
    Skp1 = Sk;
    //Get lkxx = LQR Q because it's the state cost matrix
    costJac = sat.costJacobians(k, N, xk, uk,ukp,sk,ek, bk,&costSettings_tmp);
    lkxx = costJac.lxx;
    // lkxx = mat66().eye();//costJac.lxx;
    lkuu = costJac.luu;//#*1e5; //REMOVE BEFORE FLIGHT
    // //lkxx = join_rows(join_cols(mat33().eye()*swpoint, mat33().zeros()),join_cols(mat33().zeros(),mat33().eye()*sv1));
    //Get Gk
    Gk = sat.findGMat(qk);
    //Get A, B
    tuple<mat, mat, mat> AB = rk4zJacobians(dt_tvlqr0,xk, uk, sat,dynamics_info_k,dynamics_info_kp1);
    A = get<0>(AB);//px4MatToArma(&A_px4);
    B = get<1>(AB);//px4MatToArma(&B_px4);
    C = get<2>(AB);
    //Get Aqk and Bqk
    Aqk.zeros();
    Bqk.zeros();
    Bqk.rows(0,sat.reduced_state_N()-1) = Gkp1*B;
    Aqk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) = Gkp1*A*trans(Gk);
    Aqk(span(0,sat.reduced_state_N()-1),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) = Gkp1*C;
    Aqk(span(sat.reduced_state_N(),sat.reduced_state_N()+2),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) = mat33().eye();
    //Get Kk = (R+Bqk.'*Skp1*Bqk)\(Bqk.'*Skp1*Aqk)
    //mat tmpVal = lkuu + trans(Bqk)*Skp1*Bqk;
    //eig_gen(eigvals,eigvecs,tmpVal);
    //Kk = eigvecs*diagmat(1/clamp(abs(eigvals),1e-8,datum::inf))*eigvecs.t()*trans(Bqk)*Skp1*Aqk;
    Kk = solve((lkuu + trans(Bqk)*Skp1*Bqk), (trans(Bqk)*Skp1*Aqk),solve_opts::likely_sympd+solve_opts::no_approx);//+solve_opts::refine);//inv(R + trans(Bqk)*Skp1*Bqk)*(trans(Bqk)*Skp1*Aqk);//

    Kset_lqr.slice(k) = Kk;
    // Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk = lkxx + trans(Kk)*lkuu*Kk + solve((Aqk-Bqk*Kk), Skp1*(Aqk-Bqk*Kk));
    // Sk = trans(Kk)*lkuu*Kk + trans(Aqk-Bqk*Kk)*Skp1*(Aqk-Bqk*Kk);
    Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) += lkxx;

    // if(k==N-2){cout<<Kk<<"\n";cout<<Sk<<"\n";}
    Sk = 0.5*(Sk+trans(Sk));
    // Sk = 0.5*(Sk+lkxx);
    Sset.slice(k) = Sk;
  }
  cout<<size(Kset_lqr)<<"\n";
  cout<<size(Sset)<<"\n";
  return make_tuple(Kset_lqr, Sset);
}


/*This function generates a (NOT initial) trajectory for the trajectory optimizer, using rk4, based on altering a previous trajectory
  Arguments:
    Xset - previous trajectory states - 7 x N matrix
    Uset - previous trajectory control inputs - 3 x N-1 matrix
    Kset - gain K at each timestep (from backwards pass) - 3 x 6 x N cube
    dset - from backwards pass - 3 x N matrix
    Rset - orbital position at each timestep - 3 x N matrix
    alph - double, (hyper)parameter
    lambdaSet - lambda vector - 6 x N matrix
  Returns:
    newX - new states of trajectory - 7 x N matrix
    newU - new control inputs of trajectory - 3 x N matrix
*/
 TRAJECTORY_FORM OldPlanner::generateTrajectory( double dt0,  double alpha, TRAJECTORY_FORM traj,  VECTOR_INFO_FORM vecs,  cube Kset,  mat dset, bool useDist)
{
  //Initialize newU, newX
  mat newX;
  mat Xset = get<0>(traj);
  newX.copy_size(Xset);
  newX.zeros();
  //newX.fill(datum::nan);
  mat newU;
  mat Uset = get<1>(traj);
  newU.copy_size(Uset);
  newU.zeros();

  mat newTQ;
  mat TQset = get<3>(traj);
  newTQ.copy_size(TQset);
  newTQ.zeros();
  //newU.fill(datum::nan);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);


  //Copy initial state of x to newX
  vec newXk = Xset.col(0);
  newX.col(0) = newXk;

  vec newTQk = vec(3).zeros();
  // vec4 newQk = newXk.rows(sat.quat0index(),sat.quat0index()+3);
  vec4 Qkprev = vec(4,fill::zeros);
  vec dt_timevec = get<2>(traj);

  //Initialize stuff for inside loop
  //vec delX = vec(6).zeros();
  //vec newXprev = newX.col(0);
  vec oldXprev = mat(newXk);
  vec4 oldQprev = oldXprev.rows(sat.quat0index(),sat.quat0index()+3);
  //mat Kprev = Kset.slice(0);
  //vec3 delU = vec(3).zeros();
  vec newUprev = vec(sat.control_N()).zeros();
  //vec3 oldUprev = Uset.col(0);
  //vec3 dprev = dset.col(0);

  vec3 angErr;
  vec4 quatErr;
  vec otherErr = vec(sat.reduced_state_N()-6).zeros();
  vec3 avErr = vec3().zeros();
  int N = Xset.n_cols;
  //Loop from k=1 to k=N-1 and update newU and newX
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),int(useDist));
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;

  for(int k=1; k<N; k++)
  {

    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),int(useDist));
    //Update all the "prev" variables
    //newXprev = newXk;
    oldXprev = Xset.col(k-1);
    oldQprev = oldXprev.rows(sat.quat0index(),sat.quat0index()+3);
    Qkprev =  newXk.rows(sat.quat0index(),sat.quat0index()+3);

    quatErr = normquaterr(oldQprev,Qkprev);//normalise(join_cols(vec(1).ones()*as_scalar(oldQprev.t()*Qkprev),oldQprev(0)*Qkprev.rows(1,3) - Qkprev(0)*oldQprev.rows(1,3)-cross(oldQprev.rows(1,3),Qkprev.rows(1,3))));
    // quatErr = normalise(join_cols(vec(1).ones()*as_scalar(oldQprev.t()*Qkprev),-oldQprev(0)*Qkprev.rows(1,3) + Qkprev(0)*oldQprev.rows(1,3)+cross(oldQprev.rows(1,3),Qkprev.rows(1,3))));

    // quatErr *= sign(quatErr(0));
    // cout<<"qE "<<quatErr.t()<<"\n";
    // cout<<Qkprev.t()<<"\n";
    // cout<<oldQprev.t()<<"\n";
    if(quaternionTo3VecMode >= 2 ){
      if (quaternionTo3VecMode ==2 )// Cayley
      {
        if(abs(quatErr(0))<EPSVAR){
          if(abs(quatErr(0))>0){
            quatErr(0) = EPSVAR*sign(quatErr(0));
          }
          else{
            quatErr(0) = EPSVAR;
          }
        }
        angErr = (quatErr.rows(1,3))/(quatErr(0));
      }else if(quaternionTo3VecMode ==3){//qev w/ qe0>0
        if(abs(quatErr(0))>0){
          angErr = quatErr.rows(1,3)*sign(quatErr(0));
        }
        else{angErr = quatErr.rows(1,3);}

      }else if(quaternionTo3VecMode ==4){// qev
        angErr = quatErr.rows(1,3);
      }else if(quaternionTo3VecMode ==5){ // 2xMRP
        angErr = 4.0*(quatErr.rows(1,3))/(1+quatErr(0));
      }else if(quaternionTo3VecMode ==6){ // 2xMRP with qe0>0
        if(abs(quatErr(0))>0){
          quatErr *= sign(quatErr(0));
        }
        angErr = 4.0*(quatErr.rows(1,3))/(1+quatErr(0));
      }
    }else{//mode is 0 or 1,(0 is MRP with qe0>0, 1 is MRP)
      if (quaternionTo3VecMode != 1)
      {
        if(abs(quatErr(0))>0){
          quatErr *= sign(quatErr(0));
        }
      } //qev w/ qe0>0
      angErr = 2.0*(quatErr.rows(1,3))/(1+quatErr(0));
    }

    // cout<<angErr.t()<<"\n";
    otherErr = newXk.tail(sat.reduced_state_N()-6) - oldXprev.tail(sat.reduced_state_N()-6);
    avErr = newXk.head(3) - oldXprev.head(3);
    newUprev = Uset.col(k-1) + Kset.slice(k-1)*join_cols(avErr,angErr,otherErr) + alpha*dset.col(k-1);
    // newUprev = Uset.col(k-1)+Kset.slice(k-1)*join_cols(newXk.rows(sat.avindex0(),sat.avindex0()+2)-oldXprev.rows(sat.avindex0(),sat.avindex0()+2),(oldQprev(0)*Qkprev.rows(1,3) - Qkprev(0)*oldQprev.rows(1,3)-cross(oldQprev.rows(1,3),Qkprev.rows(1,3)))/(as_scalar(oldQprev.t()*Qkprev))) + alpha*dset.col(k-1);

    for(int i = 0; i < sat.control_N(); i++)
    {
      double ucheck = newUprev(i);
      if((isnan(ucheck)||isinf(ucheck)))//||abs(ucheck)>100000000))
      {
        // cout<<"u is invalid!\n";
        // cout<<"ucheck "<<ucheck<<"\n";
        // cout<<i<<" "<<k<<"\n";
        // cout<<newUprev.t()<<"\n";
        // cout<<Uset.col(k-1).t()<<"\n";
        // cout<<Kset.slice(k-1)<<"\n";
        // cout<<join_cols(avErr,angErr,otherErr).t()<<"\n";
        // cout<<alpha<<"\n";
        // cout<<dset.col(k-1).t()<<"\n";
        // cout<<newXk.t()<<"\n";

        return make_tuple(newX.fill(datum::nan), newU.fill(datum::nan),dt_timevec,newTQ.fill(datum::nan));
      }
    }
    //get newXk using rk4
    //vec3 Bk = Bset.col(k-1);

    tuple<vec,vec> rk4out = rk4z(dt0,newXk, newUprev,sat,dynamics_info_kn1,dynamics_info_k);
    // cout<<"hi1 gT\n"<<endl;
    newXk = get<0>(rk4out);
    newTQk = get<1>(rk4out);
    newXk = sat.state_norm(newXk);

    // cout<<"hi2 gT\n"<<endl;
    // newQk = normalise(newXk.rows(sat.quat0index(), sat.quat0index()+3));
    // newXk.rows(sat.quat0index(), sat.quat0index()+3) = newQk;
    //Update newX and newU with the new state and control vector
    newX.col(k) = newXk;
    newTQ.col(k-1) = newTQk;
        // cout<<newUprev<<endl;
    newU.col(k-1) = newUprev;

    // cout<<"hi5 gT\n"<<endl;
  }
  return make_tuple(newX, newU,dt_timevec,newTQ);
}
/*This function generates the initial trajectory for the trajectory optimizer, using rk4
  Arguments:
    x0 - initial state vector - 7 x 1 vector
    Uset - control inputs for the trajectory, size 3 x N-1
    dt - time between steps of trajectory - double
  Returns:
   Xset - set of states in trajectory, size 7 x N matrix
*/
TRAJECTORY_FORM OldPlanner::generateInitialTrajectory(double dt0, vec x0, mat Uset,VECTOR_INFO_FORM vecs) {
  //Initialize newX

  int N = Uset.n_cols;
  mat Xset = mat(sat.state_N(), N).fill(datum::nan);
  mat TQset = mat(3, N).fill(datum::nan);

  // cout<<"x0 in c++ "<<x0.rows(3,6).t()<<endl;
  //Xset.fill(datum::nan);

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  vec4 qk = normalise(xk.rows(sat.quat0index(), sat.quat0index()+3));
  xk.rows(sat.quat0index(),sat.quat0index()+3) = qk;
  // cout<<"hi\n";
  Xset.col(0) = xk;

  // cout<<"xset0 in c++ "<<Xset.col(0).rows(3,6).t()<<endl;

  //Initialize stuff for inside loop
  //vec xprev = vec(x0);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  vec t = get<0>(vecs);

    // cout<<"hi\n";
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec uk = vec(sat.control_N()).zeros();
  vec3 Bk = Bset.col(0);


  //Loop from k = 1 to N-1 and fill in Xset, using rk4
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),1);

  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {
    // cout<<k<<endl;
    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),1);
    uk = Uset.col(k-1);
    //xprev = xk;
    // cout<<uk<<endl;
    // Bk = Bset.col(k-1);
    dynout = rk4z(dt0,xk, uk,sat,dynamics_info_kn1,dynamics_info_k);

    xk = sat.state_norm(get<0>(dynout));
    // cout<<xk<<endl;
    // xk.rows(3, 6)  = normalise(xk.rows(3, 6));
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);
  }

  // cout<<"xset in c++ "<<Xset.col(0).rows(3,6).t()<<endl;
  return make_tuple(Xset,Uset,t,TQset);
}



/*This method actually does alilqr
  Inputs:
    Xset - set of states so far - 7 x N matrix
    Uset - set of control inputs - 3 x N-1 matrix
    Rset - orbital position - 3 x N matrix
    Vset - orbital velocity - 3 x N matrix
    Bset - orbital magnetic field - 3 x N matrix
    R - control cost, 3x3 matrix
    QN - Q at t = N, 6x6 matrix
    costSettings - settings for Q
    forwardPassSettings - settings for forwardPass
    alilqrSettings - settings for alilqr - contains int lagMultInit, double penInit, int regInit, int maxOuterIter,
    double gradTol, double costTol, double cmax, int zCountLim, int maxIter, double penMax, double penScale, int lagMultMax,
    double ilqrCostTol
  Outputs:
    P - covariance - 6 x 6 x N cube
    K - gains - 3 x 6 x N cube
    dset - unclear - 3 x N matrix
    delV - 1 x 2 matrix
    rho,drho - same as above
*/
ALILQR_OUTPUT_FORM OldPlanner::alilqr(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, COST_SETTINGS_FORM costSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp,bool isFirstSearch)
{

  LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp = get<0>(alilqrSettings_tmp);
  AUGLAG_SETTINGS_FORM auglagSettings_tmp = get<1>(alilqrSettings_tmp);
  BREAK_SETTINGS_FORM breakSettings_tmp = get<2>(alilqrSettings_tmp);
  REG_SETTINGS_FORM regSettings_tmp = get<3>(alilqrSettings_tmp);

  double lagMultInit_tmp = get<0>(auglagSettings_tmp);
  double penInit_tmp = get<2>(auglagSettings_tmp);
  double penScale_tmp = get<4>(auglagSettings_tmp);
  double maxCost_tmp = get<8>(breakSettings_tmp);

  int maxOuterIter_tmp = get<0>(breakSettings_tmp);
  int maxIlqrIter_tmp = get<1>(breakSettings_tmp);
  int zCountLim_tmp = get<6>(breakSettings_tmp);
  double cmax_tmp = get<7>(breakSettings_tmp);

  double regInit_tmp = get<0>(regSettings);


  //double eps = this->eps;
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  vec dt_timevec = get<2>(traj);
  // cout<<dt_timevec<<endl;
  // cout<<(dt_timevec-0.22)*36525.0*24.0*3600.0<<endl;
  // cout<<dt0<<endl;

  int N = Xset.n_cols;
  //initialize grad, iter, lambdaSet, mu, LA0
  double grad = 1.0/EPSVAR;
  int iter = 0;
  mat lambdaSet = mat(sat.constraint_N(), N).ones()*lagMultInit_tmp;
  mat muSet = mat(sat.constraint_N(), N).ones()*penInit_tmp/penScale_tmp;
  double mu = penInit_tmp/penScale_tmp;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet);

  // mat clist = mat(sat.constraint_N(),N);

  traj = generateInitialTrajectory(dt0,Xset.col(0), Uset,vecs);


  tuple<mat, double> viol = OldPlanner::maxViol(traj,vecs,auglag_vals);
  mat clist = get<0>(viol);
  auglag_vals = OldPlanner::incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
  AUGLAG_INFO_FORM auglag_vals_clean = make_tuple(0*lambdaSet,0,0*muSet);
  double LA0 = OldPlanner::cost2Func(traj, vecs, auglag_vals, &costSettings_tmp);
  double LA = LA0;
  double LAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals_clean, &costSettings_tmp);;


  double cmaxtmp = 0.0;
  double dlaZcount = 0;
  double dLA = 0.0;
  double newLA = LA;
  REG_PAIR regs = make_tuple(regInit_tmp,0.0);
  //initialize Kset, Pset
  BACKWARD_PASS_RESULTS_FORM BPresults;
  tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> ilqrRes;
  //Outer loop
  if(verbose) {
    cout<<"begin outer\n";
  }
  double stepsSinceRand = -1;

  for(int j = 0; j < maxOuterIter_tmp; j++)
  {
     if(verbose){cout<<"outer iter "<<j<<"\n";}
    //reset cmaxtmp, dlaZcount
    cmaxtmp = 0.0;
    dlaZcount = 0;
    clist.zeros();
    dLA = 0.0;
    stepsSinceRand = -1;
    //ILQR
    //set rho and drho to regInit
    regs = make_tuple(regInit_tmp,0.0);
    //Find initial cost and init dLA
    LA = OldPlanner::cost2Func(traj,vecs, auglag_vals, &costSettings_tmp);
    if(verbose){
      OldPlanner::costInfo(traj, vecs, auglag_vals,&costSettings_tmp);
    }
    //inner loop
    for(int ii = 0; ii < maxIlqrIter_tmp; ii++)
    {

      //update iter
      if(verbose){cout<<"ii: "<<ii<<endl;}
      iter++;
      // tuple<TRAJECTORY_FORM,double>  rnOut = OldPlanner::addRandNoise(dt0, traj,  dlaZcount,  stepsSinceRand, breakSettings_tmp, regSettings_tmp, &costSettings_tmp, auglag_vals, vecs);
      // traj = get<0>(rnOut);
      // stepsSinceRand = get<1>(rnOut);

      ilqrRes = OldPlanner::ilqrStep(dt0,traj,vecs,auglag_vals,regs,&costSettings_tmp,regSettings_tmp,lineSearchSettings_tmp,breakSettings_tmp,!isFirstSearch);

      newLA = get<0>(ilqrRes);
      cmaxtmp = get<1>(ilqrRes);
      clist = get<2>(ilqrRes);
      grad = get<3>(ilqrRes);
      regs = get<4>(ilqrRes);
      traj = get<5>(ilqrRes);

      dLA = abs(newLA-LA);
      // if(stepsSinceRand != 0){
        dlaZcount++;
        if(dLA != 0)
        {
          dlaZcount = 0;
        }
      // }
      // if(stepsSinceRand>=0){stepsSinceRand++;}
      //update LA, Xset, Uset
      LA = newLA;
      LAnc = OldPlanner::cost2Func(traj,vecs, auglag_vals_clean, &costSettings_tmp);
      current_traj = traj;
      current_Xset = get<0>(traj);
      current_Uset = get<1>(traj);
      current_ilqr_iter = ii;
      current_outer_iter = j;
      if(verbose){cout<<ii<<" "<<j<<" cmaxtmp,LA,LA clean "<<cmaxtmp<<" "<<LA<<" "<<LAnc<<"\n";}

      //Check if we need to break out of the loop
      if(OldPlanner::ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp)){
	       // cout<<"innerbreak\n";
        break;
      }
    }
    if(OldPlanner::outerBreak(auglag_vals,cmaxtmp,breakSettings_tmp,auglagSettings_tmp)&&j>2&&OldPlanner::ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,true)) {
    // if(OldPlanner::outerBreak(auglag_vals,cmaxtmp,breakSettings_tmp,auglagSettings_tmp)){//&&OldPlanner::ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,true)) {

      cout<<"outerbreak\n";
      break;
    }
    //update lambdaSet, etc.
  // if(verbose){cout<<"auglagvals\n";}
    auglag_vals = OldPlanner::incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
  }
  if(verbose){cout<<"out of loops\n";}
  mat Kmat = packageK(get<0>(BPresults));
  if(verbose){cout<<"Kmat done\n";}
  OPT_FORM opt = make_tuple(get<0>(traj),get<1>(traj),get<3>(traj),Kmat,lambdaSet,dt_timevec);
  if(verbose){cout<<"opt packaged\n";}
  return make_tuple(opt, mu, grad);
}
tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> OldPlanner::ilqrStep(double dt0,TRAJECTORY_FORM traj,VECTOR_INFO_FORM vecs,AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs,COST_SETTINGS_FORM *costSettings_ptr,REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,BREAK_SETTINGS_FORM breakSettings_tmp,bool useDist){
  //Check if planner killed from python
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }

  if (verbose){
    OldPlanner::costInfo(traj, vecs, auglag_vals,  costSettings_ptr);
  }

  double N = get<0>(traj).n_cols;


  tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> backwardPassResults = OldPlanner::backwardPass(dt0,traj,vecs,auglag_vals,regs,costSettings_ptr,regSettings_tmp,useDist);

  BACKWARD_PASS_RESULTS_FORM BPresults = get<0>(backwardPassResults);
  regs = get<1>(backwardPassResults);
  //call forward pass

  // mat newTQ = get<3>(traj);
  // cout<<"hi00 at input\n"<<endl;
  // cout<<newTQ.n_cols<<endl;
  // cout<<newTQ.n_rows<<endl;
  // cout<<newTQ<<endl;
  tuple<TRAJECTORY_FORM, double, REG_PAIR> forwardPassOut = OldPlanner::forwardPass(dt0,traj,vecs,auglag_vals,BPresults,regs,costSettings_ptr,regSettings_tmp,lineSearchSettings_tmp,useDist);
  // cout<<"hi0\n"<<endl;
  double newLA = get<1>(forwardPassOut);
  regs = get<2>(forwardPassOut);
  traj = get<0>(forwardPassOut);
  double newLAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals, costSettings_ptr,false);
  // cout<<"hi1\n"<<endl;

  // double grad = sum( max(abs(get<1>(BPresults).cols(0,N-2))/(sqrt(sum(get<1>(traj).cols(0,N-2) % get<1>(traj).cols(0,N-2),0))+1),0))/(N-1);
  // cout<<"gradtest"<<endl;
  // cout<<abs(get<1>(BPresults).cols(0,5))<<endl;
  // cout<<(abs(get<1>(traj).cols(0,5) )+1)<<endl;
  // cout<<abs(get<1>(BPresults).cols(0,5))/(abs(get<1>(traj).cols(0,5) )+1)<<endl;
  // cout<<max(abs(get<1>(BPresults).cols(0,5))/(abs(get<1>(traj).cols(0,5) )+1),0)<<endl;
  double grad = sum( max(abs(get<1>(BPresults).cols(0,N-2))/(abs(get<1>(traj).cols(0,N-2) )+1),0))/(N-1);
  // double grad = sum( vecnorm(get<1>(BPresults).cols(0,N-2),"inf",0)/(vecnorm(get<1>(traj).cols(0,N-2) )+1))/(N-1);
  // double grad = sum( abs(get<1>(BPresults).cols(0,N-2))/(abs(get<1>(traj).cols(0,N-2) )+1))/(N-1);

  // cout<<"PROP TORQ"<<sat.prop_torq<<endl;
  // cout<<"PROP TORQ"<<sat.prop_torq<<endl;


  //find dLA
  tuple<mat, double> viol = OldPlanner::maxViol(traj,vecs,auglag_vals);
  mat clist = get<0>(viol);
  double cmaxtmp = get<1>(viol);
  if(newLAnc>get<8>(breakSettings_tmp)){
    regs = increaseReg(regs,regSettings_tmp);
  }
  return make_tuple(newLA,cmaxtmp,clist,grad,regs,traj);
}
bool OldPlanner::ilqrBreak(double grad,double LA, double dLA, double dlaZcount, double cmaxtmp, double iter,BREAK_SETTINGS_FORM breakSettings_tmp,bool forOuter)
{
  if(verbose){cout<<"ilqrBreak\n";}
  int maxOuterIter_tmp = get<0>(breakSettings_tmp);
  int maxIter_tmp = get<2>(breakSettings_tmp);
  double gradTol_tmp = get<3>(breakSettings_tmp);
  double ilqrCostTol_tmp = get<4>(breakSettings_tmp);
  double costTol_tmp = get<5>(breakSettings_tmp);
  int zCountLim_tmp = get<6>(breakSettings_tmp);
  double cmax_tmp = get<7>(breakSettings_tmp);
  double  max_cost = get<8>(breakSettings_tmp);
  // if(verbose){cout<<"unpacked breakSettings\n";}

  double useCostTol = ilqrCostTol_tmp;
  if((current_outer_iter>=maxOuterIter_tmp-1) || forOuter){
    useCostTol = costTol_tmp;
  }//
  if(verbose){cout<<"useCostTol "<<useCostTol<<"\n";}
  //if ((((cmaxtmp<cmax) || j < maxOuterIter) && grad<gradTol) ||dlaZcount > zCountLim ||(0 < dLA && dLA < ilqrCostTol && ((cmaxtmp<cmax) || j < maxOuterIter ) ))

  // (
  //   (
  //     (
  //       (cmaxtmp<cmax)
  //       ||
  //       (j<maxOuterIter)
  //     )
  //     &&
  //     (grad<gradTol)
  //   )
  //   ||
  //   (dlaZcount > zCountLim)
  //   ||
  //   (
  //     (0<=dLA)
  //     &&
  //     (dLA<ilqrCostTol)
  //     &&
  //     (
  //       (cmaxtmp<cmax)
  //       ||
  //       j < maxOuterIter
  //     )
  //   )
  // )
  // if(((grad<gradTol_tmp)||(0<=dLA && dLA<useCostTol)&&((!ls_failed)))//&&((!ls_failed)||(!forOuter)))
  //     ||((dlaZcount > zCountLim_tmp))//||((!forOuter)&&(dlaZcount > zCountLim_tmp))
  //     ||(LA>max_cost))
  if(((grad<gradTol_tmp)&&(0<dLA && dLA<useCostTol)&&(!ls_failed))||
    (dlaZcount > zCountLim_tmp) )
  {
    if(verbose) {
      cout<<"breaking inner loop alilqr with value j: "<<current_outer_iter<<" and value ii: "<<current_ilqr_iter<<"\n";
      cout<<"cmaxtmp "<<cmaxtmp<<"\n";
      cout<<"grad "<<grad<<"vs"<<gradTol_tmp<<"\n";
      cout<<"dLA "<<dLA<<"vs"<<useCostTol<<"\n";
      cout<<"line search failed?: "<<ls_failed<<"\n";
      cout<<"zcount "<<dlaZcount<<"vs"<<zCountLim_tmp<<"\n";
    }
    return true;
  }
  if(verbose){
    cout<<"checked break conditions\n";
    if(verbose) {
      cout<<" outer iter: "<<current_outer_iter<<" ilqr iter: "<<current_ilqr_iter<<" cmaxtmp: "<<cmaxtmp<<" grad: "<<grad<<" vs gradtol: "<<gradTol_tmp<<" dLA: "<<dLA<<" vs costTol: "<<useCostTol<<" zcount: "<<dlaZcount<<" vs zcountlim: "<<zCountLim_tmp<<"\n";
    }
  }
  if(iter == maxIter_tmp)
  {
    cout<<"breaking because iter == maxIter_tmp\n";
    return true;
  }
  if(iter > maxIter_tmp)
  {
    cout<<"Total iteration limit exceeded in alilqr\n";
    throw "Total iteration limit exceeded in alilqr";
  }
  if(verbose){cout<<"checked iteration limit\n";}
  return false;
}


tuple<TRAJECTORY_FORM,double>  OldPlanner::addRandNoise(double dt0, TRAJECTORY_FORM traj, double dlaZcount, double stepsSinceRand, BREAK_SETTINGS_FORM breakSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,COST_SETTINGS_FORM *costSettings_ptr, AUGLAG_INFO_FORM auglag_vals,VECTOR_INFO_FORM vecs){


  double randPercent = get<6>(regSettings_tmp);
  // if(verbose) {
  //   cout<<"trying rand??? "<<dlaZcount<<" "<<std::max(2.0,get<6>(breakSettings_tmp)*0.5)<<"\n";
  // }
  if(dlaZcount>std::max(2.0,get<6>(breakSettings_tmp)*0.5) && stepsSinceRand<0 && randPercent > 0) {
    mat Xset = get<0>(traj);
    int N = Xset.n_cols;
    mat Uset = get<1>(traj);
    // TRAJECTORY_FORM newTraj = generateInitialTrajectory(dt0,Xset.col(0), Uset.cols(0,N-2) + randPercent*diagmat(max(abs(Uset.cols(0,N-2)),1))*2*(randu(size(Uset.cols(0,N-2)))-0.5),vecs);
    mat Unoise = randPercent*abs(Uset) % (2*randu(size(Uset))-1.0);
    // if(verbose) {
    //   cout<<"trying adding random\n";
    // }
    TRAJECTORY_FORM newTraj = generateInitialTrajectory(dt0,Xset.col(0), Uset + Unoise,vecs);


    double testLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_ptr);
    if(!(isnan(testLA)||isinf(testLA)) && randPercent > 0) {
      traj = newTraj;//make_tuple(get<0>(newTraj),get<1>(newTraj),get<2>(traj),get<3>(newTraj));
      if(verbose) {
        cout<<"a bit of random added\n";
      }
      stepsSinceRand = 0;
    }

  }
  return make_tuple(traj,stepsSinceRand);
}


bool OldPlanner::outerBreak(AUGLAG_INFO_FORM auglag_vals, double cmaxtmp,BREAK_SETTINGS_FORM breakSettings_tmp,AUGLAG_SETTINGS_FORM auglagSettings_tmp)
{
  cout<<"outerBreak\n";
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);
  double cmax_tmp = get<7>(breakSettings_tmp);
  double penMax_tmp = get<3>(auglagSettings_tmp);
  if (((cmaxtmp<cmax_tmp)|| (muSet.max() >= penMax_tmp)))
  {
    if(verbose) {
      cout<<"breaking outer loop alilqr with value j: "<<current_outer_iter<<"\n";
      cout<<"cmaxtmp: "<<cmaxtmp<<"vs"<<cmax_tmp<<"\n";
      cout<<"penMax: "<<mu<<"vs"<<penMax_tmp<<"\n";
      cout<<"penMax2: "<<muSet.max()<<"vs"<<penMax_tmp<<"\n";
    }
    return true;
  }
  return false;
}

void OldPlanner::costInfo(TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, AUGLAG_INFO_FORM auglag_vals,  COST_SETTINGS_FORM *costSettings_ptr){
    // cout<<"costInfo\n";
    mat Xset = get<0>(traj);
    mat Uset = get<1>(traj);
    mat TQset = get<3>(traj);
    double N = Xset.n_cols;
    vec dt_timevec = get<2>(traj);
    mat U0 = mat(arma::size(Uset)).zeros();
    mat Unomtq = Uset;
    Unomtq.head_rows(sat.number_MTQ) *= 0;
    mat lambdaSet = get<0>(auglag_vals);
    double mu = get<1>(auglag_vals);
    mat muSet = get<2>(auglag_vals);
    mat lambda0 = mat(arma::size(lambdaSet)).zeros();
    double pen0 = 0.0;
    mat penSet0 = mat(arma::size(muSet)).zeros();

    mat satvec = get<5>(vecs);
    mat ECIvec = get<6>(vecs);
    vec angs = vec(N);
    vec ek;
    for(int k =0;k<N;k++){
      ek = ECIvec.col(k);
      if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
        ek = ek.tail(3);
        angs(k) = (180.0/datum::pi)*acos(norm_dot(satvec.col(k),rotMat(Xset(sat.quat0index(),k,size(4,1))).t()*ek));
      }else{
        angs(k) = (180.0/datum::pi)*acos(2.0*pow(norm_dot(Xset(sat.quat0index(),k,size(4,1)),ek),2.0)-1.0);
      }

    }
    vec angdiffs = diff(angs);
    vec avs = sum(square(Xset.head_rows(3))).t();
    vec avdiffs = diff(avs);
    vec hs = vec(N).zeros();
    vec hdiffs = vec(N-1).zeros();
    vec urws = vec(N).zeros();
    vec urwdiffs = vec(N-1).zeros();
    if(sat.number_RW>0){
      hs = sqrt(sum(square(Xset.tail_rows(sat.number_RW)))).t();
      hdiffs = diff(hs);
      urws = sqrt(sum(square(Uset.rows(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)))).t();
      urwdiffs = diff(urws);
    }
    vec umags = vec(N).zeros();
    vec umagdiffs = vec(N-1).zeros();
    if(sat.number_magic>0){
      umags = sqrt(sum(square(Uset.tail_rows(sat.number_magic)))).t();
      umagdiffs = diff(umags);
    }
    vec umtqs = vec(N).zeros();
    vec umtqdiffs = vec(N-1).zeros();
    if(sat.number_MTQ>0){
      umtqs = sqrt(sum(square(Uset.head_rows(sat.number_MTQ)))).t();
      umtqdiffs = diff(umtqs);
    }


      //Extract costSettings
    COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
    double angle_weight_tmp = get<0>(costSettings_tmp);
    double angvel_weight_tmp = get<1>(costSettings_tmp);
    double u_weight_tmp = get<2>(costSettings_tmp);
    double av_with_mag_weight_tmp = get<4>(costSettings_tmp);
    double ang_av_weight_tmp = get<5>(costSettings_tmp);
    double angle_weight_N_tmp = get<6>(costSettings_tmp);
    double angvel_weight_N_tmp = get<7>(costSettings_tmp);
    double av_with_mag_weight_N_tmp = get<8>(costSettings_tmp);
    double ang_av_weight_N_tmp = get<9>(costSettings_tmp);
    int whichAngCostFunc_tmp = get<10>(costSettings_tmp);
    int useRawControlCost_tmp = get<11>(costSettings_tmp);
    COST_SETTINGS_FORM nou_Settings = make_tuple(angle_weight_tmp,angvel_weight_tmp,0.0,0.0,av_with_mag_weight_tmp,ang_av_weight_tmp,angle_weight_N_tmp,angvel_weight_N_tmp,av_with_mag_weight_N_tmp,ang_av_weight_N_tmp,whichAngCostFunc_tmp,useRawControlCost_tmp);
    COST_SETTINGS_FORM only_av_Settings = make_tuple(0.0,angvel_weight_tmp,0.0,0.0,0.0,0.0,0.0,angvel_weight_N_tmp,0.0,0.0,whichAngCostFunc_tmp,useRawControlCost_tmp);
    COST_SETTINGS_FORM only_ang_Settings = make_tuple(angle_weight_tmp,0.0,0.0,0.0,0.0,0.0,angle_weight_N_tmp,0.0,0.0,0.0,whichAngCostFunc_tmp,useRawControlCost_tmp);

    mat clearvel = mat(sat.state_N(),sat.state_N()).eye();
    clearvel(span(0,2),span(0,2)).zeros();

    TRAJECTORY_FORM cleartraj = make_tuple(clearvel*Xset,U0,dt_timevec,TQset);
    TRAJECTORY_FORM noutraj = make_tuple(Xset,U0,dt_timevec,TQset);
    TRAJECTORY_FORM nomtqtraj = make_tuple(Xset,Unomtq,dt_timevec,TQset);
    AUGLAG_INFO_FORM auglag_vals_zero = make_tuple(lambda0,pen0,penSet0);

    double LA = OldPlanner::cost2Func(traj, vecs, auglag_vals, costSettings_ptr);
    double LAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals_zero, costSettings_ptr);
    double LAnou = OldPlanner::cost2Func(noutraj, vecs, auglag_vals_zero, &nou_Settings);
    double LAnomtq = OldPlanner::cost2Func(nomtqtraj, vecs, auglag_vals_zero, &nou_Settings);
    double LAang = OldPlanner::cost2Func(cleartraj,vecs,auglag_vals_zero, &only_ang_Settings);
    double LAnouav = OldPlanner::cost2Func(cleartraj,vecs,auglag_vals_zero, &nou_Settings);
    double LAav = LAnou-LAnouav;//OldPlanner::cost2Func(noutraj,vecs,auglag_vals_zero, &only_av_Settings);
    double LAu = LAnc - LAnou;
    double LAmtq = LAnc-LAnomtq;
    // double LAav = LAnc-LAu-LAang;
    double avg_ang = LAang/((N-1)*angle_weight_tmp+angle_weight_N_tmp);

    cout<<"LA: "<<LA<<" and LA w/o constraints: "<< LAnc <<"\n";
    cout<<"ang nan,max,min,mean: "<<angs.has_nan()<<" "<<angs.max()<<" "<<angs.min()<<" "<<mean(angs)<<"\n";
    cout<<"angdiff nan,max,min,mean: "<<angdiffs.has_nan()<<" "<<angdiffs.max()<<" "<<angdiffs.min()<<" "<<mean(angdiffs)<<"\n";
    cout<<"av nan,max,min,mean: "<<avs.has_nan()<<" "<<avs.max()<<" "<<avs.min()<<" "<<mean(avs)<<"\n";
    cout<<"avdiff nan,max,min,mean: "<<avdiffs.has_nan()<<" "<<avdiffs.max()<<" "<<avdiffs.min()<<" "<<mean(avdiffs)<<"\n";
    if(sat.number_RW>0){
      cout<<"h nan,max,min,mean: "<<hs.has_nan()<<" "<<hs.max()<<" "<<hs.min()<<" "<<mean(hs)<<"\n";
      cout<<"hdiff nan,max,min,mean: "<<hdiffs.has_nan()<<" "<<hdiffs.max()<<" "<<hdiffs.min()<<" "<<mean(hdiffs)<<"\n";

      cout<<"urw nan,max,min,mean: "<<urws.has_nan()<<" "<<urws.max()<<" "<<urws.min()<<" "<<mean(urws)<<"\n";
      cout<<"urwdiff nan,max,min,mean: "<<urwdiffs.has_nan()<<" "<<urwdiffs.max()<<" "<<urwdiffs.min()<<" "<<mean(urwdiffs)<<"\n";
    }
    if(sat.number_MTQ>0){
      cout<<"umtq nan,max,min,mean: "<<umtqs.has_nan()<<" "<<umtqs.max()<<" "<<umtqs.min()<<" "<<mean(umtqs)<<"\n";
      cout<<"umtqdiff nan,max,min,mean: "<<umtqdiffs.has_nan()<<" "<<umtqdiffs.max()<<" "<<umtqdiffs.min()<<" "<<mean(umtqdiffs)<<"\n";
    }
    if(sat.number_magic>0){
      cout<<"umag nan,max,min,mean: "<<umags.has_nan()<<" "<<umags.max()<<" "<<umags.min()<<" "<<mean(umags)<<"\n";
      cout<<"umagdiff nan,max,min,mean: "<<umagdiffs.has_nan()<<" "<<umagdiffs.max()<<" "<<umagdiffs.min()<<" "<<mean(umagdiffs)<<"\n";
    }
    cout<<"ang/h cost: "<<LAang<<" omega: "<<LAav<<" mtq: "<<LAmtq<<" and u: "<<LAu<<"\n";
    // cout<<"if phi cost evenly across time: "<<angest<<" omega: "<<pow(2*LAav/((N-1)*angvel_weight_tmp+angvel_weight_N_tmp),0.5)*180.0/datum::pi<<" and mtq: "<< pow(2*LAu/(u_weight_tmp*(N-1)),0.5) <<"\n";

    return;
}

AUGLAG_INFO_FORM OldPlanner::incrementAugLag(AUGLAG_INFO_FORM auglag_vals, mat clist, AUGLAG_SETTINGS_FORM auglagSettings_tmp){
    mat lambdaSet = get<0>(auglag_vals);
    double mu = get<1>(auglag_vals);
    mat muSet = get<2>(auglag_vals);

    double LMmax = get<1>(auglagSettings);
    double muMax = get<3>(auglagSettings);
    double muScale = get<4>(auglagSettings);

    double N = lambdaSet.n_cols;
    for(int k = 0; k < N; k++)
    {
      for(int i = 0; i < sat.constraint_N(); i++)
      {
        // if(clist(i,k)>0){
          lambdaSet(i,k) = lambdaSet(i,k) + muSet(i,k)*clist(i,k);
        // }
          lambdaSet(i,k) = min(LMmax*1.0, lambdaSet(i,k));
          //double minTmp = min(lagMultMax*1.0, (lambdaSet(i,k) + muSet(i,k)*max(0.0,clist(i,k))));
          lambdaSet(i, k) = max(-LMmax*1.0, lambdaSet(i,k));
          if(i < sat.ineq_constraint_N()) //because all of our constraints are limits, not equality constraints. If it was an equality constraint, we would allow it to be negative.
          {
            lambdaSet(i, k) = max(0.0, lambdaSet(i,k));
          }
          // if(clist(i,k)>=-cmax){
          // if(clist(i,k)<=cmax){
            muSet(i,k) = max(0.0,min(muMax*1.0, muScale*muSet(i,k)));
          // }
      }
    }
  //update mu
  mu = max(0.0,min(muMax*1.0, muScale*mu));
  return make_tuple(lambdaSet,mu,muSet);

}

/* This method gets the max violations and constraint list */
tuple<mat, double> OldPlanner::maxViol(TRAJECTORY_FORM &traj, VECTOR_INFO_FORM &vecs,AUGLAG_INFO_FORM &auglag)
{
  mat lambdaSet = get<0>(auglag);
  double mu = get<1>(auglag);
  mat muSet = get<2>(auglag);

  mat Uset = get<1>(traj);
  mat Xset = get<0>(traj);
  Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  int N = Xset.n_cols;
  mat sunset = get<4>(vecs);
  mat clist = mat(sat.constraint_N(), N);
  vec uk;
  vec xk;
  vec3 sunk;
  //loop over trajectory and fill in clist
  for(int k = 0; k < N; k++)
  {
    uk = Uset.col(k);
    xk = Xset.col(k);
    sunk = normalise(sunset.col(k));
    vec ck = sat.getConstraints(k, N, uk,xk,sunk);
    clist.col(k) = ck;
  }
  //clist = clamp(clist,0.0,datum::inf);
  mat corrected_clist(arma::size(clist));
  if(sat.eq_constraint_N()>0){
    corrected_clist = join_cols(clamp(clist.rows(0,sat.ineq_constraint_N()-1),0.0,datum::inf),clist.rows(sat.ineq_constraint_N(),sat.constraint_N()-1));
  }else{
    corrected_clist = clamp(clist.rows(0,sat.ineq_constraint_N()-1),0.0,datum::inf);
  }
  double cmaxtmp = abs(corrected_clist).max();

  //DEBUG
  //mat w2 = sum((Xset.rows(0,2) % Xset.rows(0,2)),0);
  uvec::fixed<2> ss=ind2sub(arma::size(corrected_clist),abs(corrected_clist).index_max());
  if(verbose){cout<<"max viol: "<<cmaxtmp<<" at subscript "<<ss(0)<<", "<<ss(1)<<"\n";}
  if(verbose){cout<<"state at max viol: "<<Xset.col(ss(1)).t()<<"\n";}
  if(verbose){cout<<"ctrl at max viol: "<<Uset.col(ss(1)).t()<<"\n";}
  if(verbose){cout<<corrected_clist.col(ss(1)).t()<<"\n";}
  // if(verbose){cout<<"violation control \n"<<muSet.col(ss(1)).t()<<"\n"<<lambdaSet.col(ss(1)).t()<<"\n";}
  // if(verbose){cout<<sat.getImu(mu, muSet.col(ss(1)), clist.col(ss(1)), lambdaSet.col(ss(1)))<<"\n";}
  // if(verbose){cout<<sat.getIlam(mu, muSet.col(ss(1)), clist.col(ss(1)), lambdaSet.col(ss(1)))<<"\n";}
  if(cmaxtmp>1e10){
    cout<<Xset.cols(0,ss(1))<<"\n";
    cout<<Uset.cols(0,ss(1))<<"\n";
  }

  return make_tuple(clist, cmaxtmp);
}

/*This method does backwardpass for AL-iLQR
  Inputs:
    Xset - set of states so far - 7 x N matrix
    Uset - set of control inputs - 3 x N-1 matrix
    Rset - orbital position - 3 x N matrix
    Vset - orbital velocity - 3 x N matrix
    Bset - orbital magnetic field - 3 x N matrix
    lambdaSet - lambda - 6 x N matrix
    rho, drho, mu, dt, regMin, regScale - various parameters, all doubles
    R - control cost, 3x3 matrix
    QN - Q at t = N, 6x6 matrix
    umax - max allowable u for constraints
    int Nslew - time at which slew turns to point - int
    vec satAlignVector - vector of satellite to align with goal - 3 x 1 vector
    qSettings - settings for Q
  Outputs:
    P - covariance - 6 x 6 x N cube
    K - gains - 3 x 6 x N cube
    dset - unclear - 3 x N matrix
    delV - 1 x 2 matrix
    rho,drho - same as above
*/
// Does qSettings need to be a pointer? Can't we use a reference?
tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> OldPlanner::backwardPass(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,bool useDist)
{
  mat Uset = get<1>(traj);
  mat Xset = get<0>(traj);
  Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  int N = Xset.n_cols;

  //Initialize return items
  cube Kset = cube(sat.control_N(), sat.reduced_state_N(), N-1).zeros();
  //cube Pset = cube(6, 6, N).zeros();
  mat dset = mat(sat.control_N(), N-1).zeros();
  vec2 delV = vec2().zeros();

  double regMin_tmp = get<1>(regSettings_tmp);
  bool useDynamicsHess_tmp = bool(get<15>(regSettings_tmp));
  bool useConstraintsHess_tmp = bool(get<16>(regSettings_tmp));

  //Initialize xk, uk, rk, etc
  mat lambdaSet = get<0>(auglag_vals);
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);

  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Bset = get<3>(vecs);
  mat sunset = get<4>(vecs);
  mat satvec = get<5>(vecs);
  mat ECIvec = get<6>(vecs);
  mat pset = get<7>(vecs);
  bool reset = false;
  double rho = 0.0;

  vec xk = vec(sat.state_N()).zeros();
  vec4 qk = vec(4).zeros();
  vec3 sunk = vec(3).zeros();
  vec dk = vec(sat.control_N()).zeros();
  mat Kk = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  //get ck (constraints)
  vec ck = vec(sat.constraint_N()).zeros();
  //get ImuK given ck
  mat Imuk = mat(sat.constraint_N(),sat.constraint_N()).zeros();
  mat Ilamk = mat(sat.constraint_N(),sat.constraint_N()).zeros();

  vec ukp = vec(sat.control_N()).zeros();
  vec viol = vec(sat.constraint_N()).zeros();
  cube violcxx = cube(sat.reduced_state_N(),sat.reduced_state_N(),sat.constraint_N()).zeros();
  cube violcux = cube(sat.control_N(),sat.reduced_state_N(),sat.constraint_N()).zeros();
  cube violcuu = cube(sat.control_N(),sat.control_N(),sat.constraint_N()).zeros();
  vec pk = vec(sat.reduced_state_N()).zeros();
  mat Pk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();

  mat Gk = mat(sat.reduced_state_N(), sat.state_N()).zeros();
  mat Gkp1 = mat(sat.reduced_state_N(), sat.state_N()).zeros();

  tuple<mat, mat,mat> AB;
  tuple<cube, cube, cube> hesses;
  mat Aqk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  mat Bqk = mat(sat.reduced_state_N(),sat.control_N()).zeros();
  cube ddxd__dxdx = cube(sat.state_N(),sat.state_N(),sat.state_N()).zeros();
  cube ddxd__dudx = cube(sat.control_N(),sat.state_N(),sat.state_N()).zeros();
  cube ddxd__dudu = cube(sat.control_N(),sat.control_N(),sat.state_N()).zeros();
  cube ddxd__dxdxQ = cube(sat.reduced_state_N(),sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  cube ddxd__dudxQ = cube(sat.control_N(),sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  cube ddxd__duduQ = cube(sat.control_N(),sat.control_N(),sat.reduced_state_N()).zeros();

  cost_jacs costJac;
  tuple<mat, mat> cnstrJac;
  tuple<cube,cube,cube> cnstrHess;

  mat cku = mat(sat.constraint_N(),sat.control_N()).zeros();
  mat ckx = mat(sat.constraint_N(),sat.reduced_state_N()).zeros();
  mat Qkuu = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkuureg = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkuureg_chol = mat(sat.control_N(),sat.control_N()).zeros();
  umat Qkuureg_chol_piv;// = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkux = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  mat Qkxx = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  vec Qku = vec(sat.control_N()).zeros();
  vec Qkx = vec(sat.reduced_state_N()).zeros();

  vec eigs = vec(sat.control_N()).zeros();
  vec modeig = vec(sat.control_N()).zeros();

  cx_vec cxeigs = cx_vec(vec(eigs),vec(eigs));
  vec eigsreg = vec(sat.control_N()).zeros();
  mat eigvecs = mat(sat.control_N(),sat.control_N()).zeros();

  DYNAMICS_INFO_FORM dynamics_info_k;
  DYNAMICS_INFO_FORM dynamics_info_kp1;
  // dynamics_info_k = make_tuple(Bset.col(N-1),Rset.col(N-1),prop_torq*plan_for_prop);
  //looping back from k = N-1 to 0:
  double regComp;
  double EVreg;
  double regAddComp;
  vec ek;
  int k = N-1;

  while(k >= 0)
  {
    //store Gk, Pk, pk
    Gkp1 = Gk;
    dynamics_info_kp1 = dynamics_info_k;
    xk = Xset.col(k);
    qk = xk.rows(3, 6);
    sunk = normalise(sunset.col(k));
    //use rk4 to find Ak, Bk
    Gk = sat.findGMat(qk);
    dynamics_info_k = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),int(useDist));
    //rk4Jacobians
    //find Aqk, Bqk
    Aqk = Aqk.zeros();
    Bqk = Bqk.zeros();
    ukp = ukp.zeros();
    if(k<N-1)
    {
      AB = rk4zJacobians(dt0,xk, Uset.col(k),sat,dynamics_info_k, dynamics_info_kp1);
      Aqk = Gkp1*get<0>(AB)*trans(Gk);
      Bqk = Gkp1*get<1>(AB);
      if(useDynamicsHess_tmp){
        hesses = rk4zHessians(dt0,xk, Uset.col(k),sat,dynamics_info_k, dynamics_info_kp1);
        ddxd__dxdx = get<0>(hesses);
        ddxd__dudx = get<1>(hesses);
        ddxd__dudu = get<2>(hesses);
        ddxd__dxdxQ = matOverCube(Gkp1,matTimesCube(Gk,cubeTimesMat(ddxd__dxdx,trans(Gk))));
        ddxd__dudxQ = matOverCube(Gkp1,cubeTimesMat(ddxd__dudx,trans(Gk)));
        ddxd__duduQ =  matOverCube(Gkp1,ddxd__dudu);
      }
    }
    //
    if (k>0){ukp = Uset.col(k-1);}else{ukp = Uset.col(0);}
    ek = ECIvec.col(k);
    if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
      ek = ek.tail(3);
      costJac = sat.veccostJacobians(k, N, xk, Uset.col(k), ukp,satvec.col(k),ek,Bset.col(k), costSettings_tmp);
    }else{
      costJac = sat.quatcostJacobians(k, N, xk, Uset.col(k), ukp,satvec.col(k),ek,Bset.col(k), costSettings_tmp);
    }

    cnstrJac = sat.constraintJacobians(k, N,Uset.col(k), xk,sunk);


    cku = get<0>(cnstrJac);
    ckx = get<1>(cnstrJac);
    ck = sat.getConstraints(k, N, Uset.col(k), xk,sunk);
    //update Imuk
    Imuk = sat.getImu(mu, muSet.col(k), ck, lambdaSet.col(k));
    Ilamk = sat.getIlam(mu, muSet.col(k), ck, lambdaSet.col(k));


    viol = (Ilamk*lambdaSet.col(k)+Imuk*ck);
    if(useConstraintsHess_tmp){
      cnstrHess = sat.constraintHessians(k, N, Uset.col(k),xk,sunk);//join_cols(mat33().eye(), -1*mat33().eye());
      // for(int i = 0; i < sat.constraint_N(); ++i)
      // {
      //   violcxx.slice(i) = mat(sat.reduced_state_N(),sat.reduced_state_N(),fill::ones)*viol(i);
      //   violcux.slice(i) = mat(sat.control_N(),sat.reduced_state_N(),fill::ones)*viol(i);
      //   violcuu.slice(i) = mat(sat.control_N(),sat.control_N(),fill::ones)*viol(i);
      // }
    }

    Qkxx = costJac.lxx + trans(Aqk)*Pk*Aqk + trans(ckx)*Imuk*ckx ;
    Qkx = costJac.lx + trans(Aqk)*pk + trans(ckx)*viol;
    if(useDynamicsHess_tmp){
      Qkxx += vecOverCube(pk,ddxd__dxdxQ);
    }
    if(useConstraintsHess_tmp){
      Qkxx += vecOverCube(viol,get<2>(cnstrHess));//mat(sum(get<2>(cnstrHess) % violcxx,2));
    }
    // Qkxx_full = costJac.lxx + trans(Aqk)*Pk*Aqk + trans(dAqkdx)*pk + mat(sum(get<2>(cnstrHess) % violcxx,2)) + trans(ckx)*Imuk*ckx;
    if(k==N-1){
      pk = Qkx;
      Pk = Qkxx;
      Pk = 0.5*(Pk + trans(Pk));
      k--;
      continue;
    }
    Qkux = costJac.lux + trans(Bqk)*Pk*Aqk + trans(cku)*Imuk*ckx;
    Qku = costJac.lu + trans(Bqk)*pk + trans(cku)*viol;

    //find Qkuu and Qkuureg
    Qkuu = costJac.luu + trans(Bqk)*Pk*Bqk + trans(cku)*Imuk*cku;

    if(useDynamicsHess_tmp){
      Qkux += vecOverCube(pk,ddxd__dudxQ);
      Qkuu += vecOverCube(pk,ddxd__duduQ);
    }
    if(useConstraintsHess_tmp){
      Qkux += vecOverCube(viol,get<1>(cnstrHess));//mat(sum(get<1>(cnstrHess) % violcux,2));
      Qkuu += vecOverCube(viol,get<0>(cnstrHess));//mat(sum(get<0>(cnstrHess) % violcuu,2));
    }

    rho = get<0>(regs);
    reset |= (Qkuu.has_nan()||Qkuu.has_inf());
    if(verbose&&reset){
      cout<<"Qkuu has nan or inf: "<<Qkuu.has_nan()<<" "<<Qkuu.has_inf()<<"\n";
    }
    if(!reset){

        // reset |= !Qkuu.is_symmetric();
        Qkuu = 0.5*(Qkuu+Qkuu.t());//trimatu(Qkuu)+trans(trimatu(Qkuu,1));
        // if(!reset){
        //   // reset |= !eig_sym(eigs,eigvecs,Qkuu);//eigs = eig_sym(Qkuu);
        //   reset |= !eig_sym(eigs,eigvecs,Qkuu);
        //   reset |= (min(eigs) <= -rho);
        // }
      // if(useDynamicsHess_tmp || useConstraintsHess_tmp){
      //   reset |= Qkuu.is_sympd()
      //   reset |= !eig_gen(cxeigs,Qkuu);
      //   reset |= (min(real(cxeigs)) < -rho);
      // }else{
      //   Qkuu = 0.5*(Qkuu+Qkuu.t());//trimatu(Qkuu)+trans(trimatu(Qkuu,1));
      //   reset |= !eig_sym(eigs,Qkuu);//eigs = eig_sym(Qkuu);
      //   reset |= (min(eigs) < -rho);
      // }
//
      // eig_sym(eigs,eigvecs,Qkuureg);
      // eig_sym(eigs,eigvecs,Qkuu);
      // Qkuureg = eigvecs*diagmat(clamp(eigs,rho,datum::inf))*eigvecs.t();
      // Qkuureg = eigvecs*diagmat(clamp(abs(eigs),rho,datum::inf))*eigvecs.t();
      Qkuureg = Qkuu + rho*mat(sat.control_N(),sat.control_N()).eye();
      // if(verbose){cout<<"regadded\n";}





      //regularization to all
      // modeig = eigs+rho; //pure inverse of Qkuu+rho*eye
      // modeig = abs(eigs)+rho; //pure inverse but abs(eigs)
      // modeig = rho+clamp(eigs,0,datum::inf);  //pure inverse but first negative eigs are eliminated.

      //clamping
      // modeig = clamp(eigs,rho,datum::inf);  //all eigs greater than rho
      // modeig = clamp(abs(eigs),rho,datum::inf);


      //unlcear
      // modeig = clamp(eigs+rho,rho,datum::inf);
      // modeig = clamp(eigs+rho,0,datum::inf);


      // reset |= (min(modeig) <= 0);
      // Qkuureg_chol = eigvecs*diagmat(1.0/modeig)*eigvecs.t();

      // Qkuureg_chol = eigvecs*diagmat(clamp(1.0/(eigs+rho),0,1.0/rho))*eigvecs.t();
      // Qkuureg_chol = eigvecs*diagmat(clamp(1.0/abs(eigs+rho),0,datum::inf))*eigvecs.t();
      // Qkuureg_chol = eigvecs*diagmat(1.0/clamp(eigs+rho,rho,datum::inf))*eigvecs.t();
      // eigsreg = eigs + rho;
      // reset |= !chol(Qkuureg_chol,Qkuureg_chol_piv,Qkuureg,"lower","matrix"); //cheap check for positive-definiteness
      reset |= !chol(Qkuureg_chol,Qkuureg);//,"lower","matrix"); //cheap check for positive-definiteness

      reset |= (Qkuureg_chol.has_nan()||Qkuureg_chol.has_inf());
      if(verbose&&reset){
        cout<<"Qkuu_reg not PD!\n";//has eigs<=-rho: "<<min(eigs)<<" < "<<-rho<<"\n";
        cout<<"k "<<k<<"\n";
        cout<<Qkuu<<"\n";
        cout<<Qku.t()<<"\n";
      }
      if(!reset){
        if(Qkuureg.has_nan()||Qkuureg.has_inf()){
          cout<<"noreg "<<Qkuu<<"\n";
          cout<<"reg "<<Qkuureg<<"\n";
          cout<<"rho "<<rho<<"\n";
          cout<<"eigs "<<eigs<<"\n";
          cout<<"cxeigs "<<cxeigs<<"\n";
          // cout<<k<<" "<<ck<<" "<<Pk<<" "<<costJac.luu<<" "<<cku<<" "<<Imuk<<" "<<Bqk<<"\n";
          cout<<"somehow regularized Qkuu has nan/inf but nonregularized does not\n";
          throw("somehow regularized Qkuu has nan/inf but nonregularized does not");
        }
        reset |= !solve(Kk,Qkuureg, Qkux,solve_opts::no_approx);//+solve_opts::likely_sympd);//+solve_opts::fast);//+solve_opts::refine);%+solve_opts::no_approx);//+solve_opts::no_approx);//+solve_opts::equilibrate+solve_opts::refine+solve_opts::no_approx);


        // if(useDynamicsHess_tmp || useConstraintsHess_tmp){
        //   reset |= !inv(Qkuureg_chol,Qkuureg,inv_opts::no_ugly);
        // }else{
        //   reset |= !inv_sympd(Qkuureg_chol,Qkuureg,inv_opts::no_ugly);
        // }
  //

        // Kk = Qkuureg_chol*Qkux;
        if(verbose&&reset){
          cout<<"Solving Kk failed \n";
        }
        reset |= !solve(dk,Qkuureg, Qku,solve_opts::no_approx);//+solve_opts::likely_sympd);//solve_opts::fast);//+solve_opts::refine+solve_opts::no_approx);//+solve_opts::no_approx);//+solve_opts::equilibrate+solve_opts::refine+solve_opts::no_approx);
        // dk = Qkuureg_chol*Qku;
        if(verbose&&reset){
          cout<<"Solving dk failed \n";
        }
        reset |= (dk.has_nan()||dk.has_inf());
        reset |= (Kk.has_nan()||Kk.has_inf());
      }
    }
    if(!reset){
      dk *= -1;
      Kk *= -1;
      Kset.slice(k) = Kk;
      dset.col(k) = dk;
      pk = Qkx + trans(Kk)*Qkuu*dk + trans(Kk)*Qku + trans(Qkux)*dk;
      Pk = Qkxx + trans(Kk)*Qkuu*Kk + trans(Kk)*Qkux + trans(Qkux)*Kk;
      if(Pk.has_nan()||Pk.has_inf()){
        cout<<"Costjac "<<costJac.lxx<<"\n";
        for(int j = 0;j<sat.number_RW;j++){

          double z = xk(7+j);
          double sz = sign(z);
          cout<<shifted_softplus(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
          cout<<shifted_softplus_deriv(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
          cout<<shifted_softplus_deriv2(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
          double d = z*sz - sat.RW_AM_cost_threshold.at(j);
          double bb = 1e6;
          cout<<(1.0+exp((d)*bb))<<"\n";
          cout<<log(1.0+exp((d)*bb))<<"\n";
          cout<<1.0/(1.0+exp(-1.0*(d)*bb))<<"\n";
          cout<<exp((d)*bb)<<"\n";
          cout<<exp(-1.0*(d)*bb)<<"\n";

          cout<<smoothstep_deriv((sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j))<<"\n";
          cout<<smoothstep((sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j))<<"\n";
          cout<<smoothstep_deriv2((sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j))<<"\n";
        }
        cout<<"Pk has nan/inf\n";
        // if(!(Pk.has_nan()||Pk.has_inf())){
        //   cout<<"somehow Qkuu has nan/inf but Pk does not\n";
        //   throw("somehow Qkuu has nan/inf but Pk does not");
        // }
        reset = true;
      }
    }
    if (reset)
    {
      k = N-1;
      delV.zeros();
      Gk.zeros();
      Pk.zeros();
      pk.zeros();
      Kset.zeros();
      dset.zeros();
      dk.zeros();
      Kk.zeros();
      reset = false;
      regs = increaseReg(regs,regSettings_tmp);
      continue;
    }
    Pk = 0.5*(Pk + trans(Pk));
    delV += join_cols(trans(dk)*Qku, 0.5*trans(dk)*Qkuu*dk);//*(get<0>(regs)+mean(eigs))/mean(eigs);//delV_int;
    dk.zeros();
    Kk.zeros();
    k--;
  }
  //find rho and drho
  regs = decreaseReg(regs,regSettings_tmp);
  return make_tuple(make_tuple(Kset,dset,delV), regs);
}
/*This function is forward pass!!
  Arguments:
    Xset - states of previous trajectory, 7 x N mat
    Uset - control vectors of previous trajectory, 3 x N-1 mat
    Kset - K gain matrices from backwards pass, 6 x 3 x N mat
    dset - from previous backwards pass, 3 x N mat
    delV - from previous backwards pass, 1 x 2 mat
    LA - cost of previous trajectory - double
    lambdaSet - lambda - 6 x N mat
    rho,drho,mu - various params
    Rset - orbital position - 3 x N mat
    Vset - orbital velocity - 3 x N mat
    QN - matrices for Q function, 3x3 mat and 6x6 mat respectively
    costSettings - settings for finding Q matrix
    forwardPassSettings - contains maxLsIter beta1 beta2 regScale regMin regBump umax xmax epsilon vNslew satAlignVector (all parameters or max constr or parameters for
      finding cost wrt alignment vectors)
  Returns:
    newX - new trajectory - 7 x N
    newU - new control states - 3 x N
    newLA - new cost - double
    rho,drho - updated params (doubles)
*/
tuple<TRAJECTORY_FORM,double, REG_PAIR> OldPlanner::forwardPass(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, AUGLAG_INFO_FORM auglag_vals, BACKWARD_PASS_RESULTS_FORM BPresults, REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp_ptr, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,bool useDist)
{
  int maxLsIter_tmp = get<0>(lineSearchSettings_tmp);
  double beta1_tmp = get<1>(lineSearchSettings_tmp);
  double beta2_tmp = get<2>(lineSearchSettings_tmp);

  cube Kset = get<0>(BPresults);
  mat dset = get<1>(BPresults);
  vec2 delV = get<2>(BPresults);

  //Get N
  mat Xset = get<0>(traj);
  int N = Xset.n_cols;
  //Initialize newU, newX
  mat newX = mat(sat.state_N(), N).zeros();
  mat newU = mat(sat.control_N(), N).zeros();
  //Initialize cost, alpha, z, exp
  double alph = 1.0;
  double newLA = 1.79769e+308;//1/eps;
  double z = -1.0;
  int lsiter = 0;
  double exp = 0.0;

  TRAJECTORY_FORM newTraj = traj;
  bool everythingOK = true;

  double LA = cost2Func(traj,vecs,auglag_vals, costSettings_tmp_ptr);
  // newTraj = generateTrajectory(dt0,0.0,traj,vecs, Kset, dset,useDist);
  // newX = get<0>(newTraj);
  // newU = get<1>(newTraj);
  // newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);;

  if(verbose){cout<<LA<<" "<<newLA<<"\n";}//here overall
  ls_failed = false;
  //Loop while z is NOT between beta2 and beta1, and the new cost is higher than the original cost
  // cout<<dset.t()<<"\n";
  if(verbose){cout<<delV.t()<<"\n";}

  newTraj = generateTrajectory(dt0,0,traj,vecs, Kset, dset,useDist);
  newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);
  if(verbose){cout<<"ls iter, LA-nLA, nLA,TEST, z,alph,reg "<<-1<<" "<<LA-newLA<<" "<<newLA<<" "<<0<<" "<<nan("1")<<" "<<0<<" "<<get<0>(regs)<<"\n";}
  newLA = 1.79769e+308;//1/eps;

  while((z<=beta1_tmp||z>beta2_tmp)||(newLA>=LA))
  {

    //If iter > maxLsIter, need to give up and just return the original trajectory if we haven't found a better new one
    if(lsiter > maxLsIter_tmp)
    {
      //double drho0 = get<1>(regs);
      regs = increaseReg(regs,regSettings_tmp);
      //increase regularization, so a second try does better
      //regs = increaseReg(regs,regSettings_tmp);
      //bump regularization even more
      regs = make_tuple(get<0>(regs) + get<4>(regSettings_tmp),get<1>(regs));//1.0/get<3>(regSettings_tmp));

      regs = increaseReg(regs,regSettings_tmp); //do 2 increases otherwise the backwardpass just undoes it
      cout<<"*************** z denied\n";
      ls_failed = true;
      return make_tuple(traj,LA, regs);

    }
    //Call generateTrajectory to get a new trajectory
    newTraj = generateTrajectory(dt0,alph,traj,vecs, Kset, dset,useDist);
    // cout<<"dmax "<<abs(dset).max()<<" "<<abs(alph*dset).max()<<endl;
    newX = get<0>(newTraj);
    newU = get<1>(newTraj);
    // cout<<newX.col(5).t()<<" "<<newU.col(5).t()<<endl;

    //If we have violated the constraints, or there was an error causing NaN in some matrix, we need to skip to the next iteration with updated iter and alph
    //Check newX for issues
    //Not sure if correctly checking for constraint violations
    // everythingOK = ;
    z = -1.0;
    if(!(newX.has_nan()||newX.has_inf()||newU.has_nan()||newU.has_inf())){//||(abs(newX).max()>100000000.0))) {
      newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);
      double newLAtest = cost2Func(traj,vecs,auglag_vals, costSettings_tmp_ptr);
      // cout<<"newLA and old LA and test"<<newLA<<" "<<LA<<" "<<newLAtest<<"\n";
        // if(verbose){cout<<"calced new LA\n";}
      // everythingOK |= !isnan(newLA);
      if(isnan(newLA)||isinf(newLA)){
        newTraj = traj;
        newLA = LA;
        cout<<"newLA is nan\n";
        lsiter++;
        alph /= 2.0;
        continue;
      }
    }
    // else{cout<<" issue with trajectory!\n";
    //     cout<<newX.has_nan()<<endl;
    //     cout<<newX.has_inf()<<endl;
    //     cout<<newU.has_nan()<<endl;
    //     cout<<newU.has_inf()<<endl;
    //     cout<<(abs(newX).max()>100000000.0)<<endl;
    //   }
      //Update exp
    exp = -alph*(delV(0) + alph*delV(1));
    // if((exp > 0.0)&&(exp<1.0e-14))
    // {
    //   //double drho0 = get<1>(regs);
    //   regs = increaseReg(regs,regSettings_tmp);
    //   //increase regularization, so a second try does better
    //   //regs = increaseReg(regs,regSettings_tmp);
    //   //bump regularization even more
    //   cout<<"*************** z denied due to small expected\n";
    //   return make_tuple(traj,LA, regs);
    //
    // } else
    if (exp > 0.0)
    {
      z = (LA-newLA)/exp;
    }

    if(verbose){cout<<"ls iter, LA-nLA, nLA, exp, z,alph,reg "<<lsiter<<" "<<LA-newLA<<" "<<newLA<<" "<<exp<<" "<<z<<" "<<alph<<" "<<get<0>(regs)<<"\n";}
    lsiter++;
    alph /= 2.0;
  }
  //If we somehow increased the cost, we need to throw an exception. This is not supposed to happen! (because we will just keep the old traj if can't find a better one)
  if(newLA > LA)
  {
    cout<<"Increased cost in forwardpass\n";
    throw("Increased cost in forwardpass");
  }
  if(verbose){cout<<"*************** z "<<z<<"\n";}
  return make_tuple(newTraj, newLA, regs);
}

/*This is the cost function, which finds the cost over a preset trajectory
  Arguments:
   Xset - preset states of trajectory, 7 x N
   Uset - preset control inputs of trajectory, 3 x N-1
   Vset - preset orbit velocity of trajectory, 3 x N
   Rset - preset orbital position of trajectory, 3 x N
   lambdaSet - preset lambda vector, 6 x N
   mu - parameter - double
   dt0 - time between steps in trajectory - double
   QN - Q function at final timestep - 6x6 matrix
   R - control cost - 3 x 3 matrix
   umax_ptr - maximum allowable u, for constraints - 3 x 1 vector
   costSettings - settings for finding Q function -- tuple<int, double, double, mat, double, double> costSettings contains: an  which describes the time to go from slew mode to pointing mode,
   double sv1, a scaling factor for the quaternion cost in pointing mode, double swpoint, a scaling factor for the
   rotation rate cost in pointing mode, rNslew, the r_ECI at time Nslew, swslew,
   a scaling factor for the rotation rate cost in slew mode, and sratioslew, a scaling factor for the overall Q
   matrix in slew mode
  Returns:
  LA - vector of cost at each timestep - double
*/
 double OldPlanner::cost2Func( TRAJECTORY_FORM &traj,  VECTOR_INFO_FORM &vecs,  AUGLAG_INFO_FORM &auglag_vals,  COST_SETTINGS_FORM *costSettings_ptr,bool useConstraints)
{
  mat Xset = get<0>(traj);
  mat Uset0 = get<1>(traj);
  mat Uset = mat(Uset0);
  if(Uset.n_cols<Xset.n_cols){
    Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  }

  COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
  int N = Xset.n_cols;
  mat sunset = get<4>(vecs);
  mat Bset = get<3>(vecs);
  mat satvec = get<5>(vecs);
  mat ECIvec = get<6>(vecs);
  mat lambdaSet = get<0>(auglag_vals);
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);
  vec ck;
  vec xk;
  vec uk;
  vec ukp;
  vec lamk;
  vec muk;
  vec3 bk;
  vec ek;
  vec3 sunk;
  vec3 sk;

  mat Imuk;
  mat Ilamk;
  double dLA = 0.0;

  double LA = 0.0;
  ukp = Uset.col(0);
  for(int k = 0; k < N; k++)
  {
    // cout<<"cost func, "<<k<<"of"<<N<<endl;
    xk = Xset.col(k);
    // cout<<1<<"\n"<<endl;
    uk = Uset.col(k);
    // cout<<2<<"\n"<<endl;
    bk = Bset.col(k);
    // cout<<3<<"\n"<<endl;
    sunk = normalise(sunset.col(k));
    // cout<<4<<"\n"<<endl;
    sk = satvec.col(k);
    // cout<<5<<"\n"<<endl;
    ek = ECIvec.col(k);
    // cout<<6<<"\n"<<endl;
    //Update ck and Imuk
    if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
      ek = ek.tail(3);
      dLA = sat.stepcost_vec(k, N, xk,uk, ukp,sk, ek,bk, costSettings_ptr);
    }else{
      dLA = sat.stepcost_quat(k, N, xk,uk, ukp,sk, ek,bk, costSettings_ptr);
    }
    // cout<<"7\n"<<endl;
    LA += dLA;
    if(isinf(dLA) || isnan(dLA)){
      cout<<dLA<<"\n";
      cout<<xk.t()<<"\n";
      cout<<uk.t()<<"\n";
      cout<<bk.t()<<"\n";
      cout<<sk.t()<<"\n";
      cout<<ek.t()<<"\n";
      cout<<k<<"\n";
      if(k>0){
        cout<<"prev\n";
        cout<<Xset.col(k-1).t()<<"\n";
        cout<<Uset.col(k-1).t()<<"\n";
        cout<<Bset.col(k-1).t()<<"\n";
        // cout<<cross(Uset.col(k-1).t(),rotMat(Xset.col(k-1).rows(3,6)).t()*Bset.col(k-1)).t()<<"\n";
        // cout<<sat.invJcom_noRW*cross(Uset.col(k-1).t(),rotMat(Xset.col(k-1).rows(3,6)).t()*Bset.col(k-1)).t()<<"\n";
      }
      for(int j = 0;j<sat.number_RW;j++)
      {
        double z = xk(7+j);
        double sz = sign(z);
        cout<<z*sz<<" "<<sat.RW_AM_cost_threshold.at(j)<<" "<<z*sz-sat.RW_AM_cost_threshold.at(j)<<" "<<1e6*(z*sz-sat.RW_AM_cost_threshold.at(j))<<" "<<exp(1e6*(z*sz-sat.RW_AM_cost_threshold.at(j)))<<"\n";
        cout<<0.5*sat.RW_AM_cost.at(j)*pow(shifted_softplus(z*sz,sat.RW_AM_cost_threshold.at(j)),2)<<"\n";
        cout<< 0.5*sat.RW_stiction_cost.at(j)*pow(smoothstep((sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j))*sat.RW_stiction_threshold.at(j),2)<<"\n";
      }

      cout<<"broken, infinite/nan cost on step.\n";
      // throw("broken, infinite/nan cost on step.");
      break;
    }

    // cout<<"8\n"<<endl;

    //assert (!isnan(stepcost_vec(k, N, xk,uk,  sk, ek,bk, costSettings_ptr)) || !(std::cerr<<k<<" "<<N<<" "<<xk.t()<<" "<<uk.t()<<" "<<sk.t()<<" "<<ek.t()<<" "<<bk.t()<<" "<<"\n"));
    if(useConstraints){
      lamk = lambdaSet.col(k);
      muk = muSet.col(k);
      ck = sat.getConstraints(k, N, uk, xk,sunk);
      Ilamk = sat.getIlam(mu, muk, ck, lamk);
      Imuk = sat.getImu(mu, muk, ck, lamk);
      LA += as_scalar(trans(lamk)*Ilamk*ck+trans(0.5*Imuk*ck)*ck);
    }

    // cout<<"9\n"<<endl;
    ukp = uk;
  }
  // cout<<"cost func done\n"<<endl;
  return LA;
}

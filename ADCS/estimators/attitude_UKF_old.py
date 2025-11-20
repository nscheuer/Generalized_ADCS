__all__ = ["UKF"]

import numpy as np
import copy
import scipy
from typing import List

from ADCS.estimators.estimator import Estimator
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import SunSensor, SunPair
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import CG5Coefficients
from ADCS.helpers.math_helpers import quat_to_vec3, vec3_to_quat, quat_mult, quat_inv, normalize, state_norm_jac, matrix_row_normalize

class extra:
    def __init__(self):
        pass

class UKF(Estimator):
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        J2000: float,
        x_hat: np.ndarray,
        P_hat: np.ndarray,
        Q_hat: np.ndarray,
        dt: float = 1.0,
        cross_term: bool = False,
        quat_as_vec: bool = False,
    ) -> None:
        """
        Unscented Kalman Filter with an error-state attitude representation.

        - self.x_hat.val is always the FULL state: [w(3), q(4), rest].
        - self.x_hat.cov is the covariance of the REDUCED error state:
            * quat_as_vec == False: [dw(3), dtheta(3), drest]  (dim = N-1)
            * quat_as_vec == True : [dw(3), dq(4),     drest]  (dim = N)
        """
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
            quat_as_vec=quat_as_vec,
        )
        self.al = 0.001
        self.kap = 0.0
        self.bet = 2.0#-1.0#2.0
        self.vec_mode = 6

    def make_pts_and_wts(self,pt0,which_sensors):
        state_cov = self.x_hat.cov.copy()
        int_cov = self.x_hat.int_cov.copy()*0.0
        control_cov = self.est_sat.control_cov()
        sens_cov = self.est_sat.sensor_cov()*0.0

        include_cov = [True,False,True,False]
        covs = [state_cov,sens_cov,control_cov,int_cov]
        zeros = [pt0,sens_cov[0,:]*0,control_cov[0,:]*0,int_cov[0,:]*0]

        L = np.sum([include_cov[j]*np.size(covs[j],0) for j in range(4)])

        lam = self.al**2.0*(self.kap+L)-L#3#3-L#self.al**2.0*(k+L)-L

        # lam = self.al**2.0*(self.kap)-L#3#3-L#self.al**2.0*(k+L)-L
        offsets = [0,0,0,0]
        pts = [zeros]
        self.scale = L+lam

        for j in range(4):
            if include_cov[j]:
                mat = np.linalg.cholesky(self.scale*covs[j])

                offsets[j] = np.hstack([mat,-mat]).T
                if j == 0:
                    states = self.add_to_state(pt0,offsets[0])
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in states]
                else:
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in offsets[j]]

        wts_m = np.array([lam/(L+lam)]+[0.5/(L+lam) for j in range(2*L)])
        wts_c = np.array([lam/(L+lam) + (1.0-self.al**2.0 + self.bet)]+[0.5/(L+lam) for j in range(2*L)])
        self.wts_m = wts_m
        self.wts_c = wts_c

        return L,pts,wts_m,wts_c,np.vstack([pt0,states]+[pt0]*(2*L-states.shape[0]))

    def reunite_states(self,dynstate,rest_state,quatref):
        if self.quat_as_vec:
            return np.concatenate([dynstate,rest_state])
        else:
            quatdiff = quat_mult(quat_inv(quatref),dynstate[3:7])
            v3diff = quat_to_vec3(quatdiff,self.vec_mode)
            return np.concatenate([dynstate[0:3],v3diff,rest_state])
        
    def add_to_state(self,state,add):
        add = np.squeeze(add)
        state = np.squeeze(state)
        if add.ndim == 1:
            if self.quat_as_vec:
                result = state+add
                result[3:7] = normalize(result[3:7])
            else:
                result = state.copy()
                result[0:3] = state[0:3] + add[0:3]
                result[7:] = state[7:] + add[6:]
                result[3:7] = quat_mult(state[3:7],vec3_to_quat(add[3:6],self.vec_mode))
        else:
            if self.quat_as_vec:
                result = state+add
                result[3:7] = matrix_row_normalize(result[3:7])
            else:
                result = np.zeros((np.size(add,0),np.size(state,0)))
                result[:,0:3] = state[0:3] + add[:,0:3]
                result[:,7:] = state[7:] + add[:,6:]
                result[:,3:7] = np.vstack([quat_mult(state[3:7],vec3_to_quat(add[j,3:6],self.vec_mode)) for j in range(np.size(add,0))])
        return result

    def new_post_state(self,pre_rest_state,post_dynstate,int_err,quatref):
        post_dyn_state_w_int_err = self.add_to_state(post_dynstate,int_err[0:self.est_sat.state_len - 1 + self.quat_as_vec])
        post_state = self.reunite_states(post_dyn_state_w_int_err,pre_rest_state+int_err[self.est_sat.state_len - 1 + self.quat_as_vec:],quatref)
        s0len = np.zeros(np.size(post_state) + 1 - self.quat_as_vec)
        s0len[3:7] = quatref
        full_state = self.add_to_state(s0len,post_state)#these are backwards on purpose
        return post_state,full_state
    
    def sat_match(self,est_sat: EstimatedSatellite,state):
        full_statej = self.x_hat.copy()
        full_statej.val[self.use] = state
        est_sat.match_estimate(full_statej,self.dt)

    def update_core(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> EstimatedArray:
        u = np.copy(u)
        os = os.copy()

        state0 = self.x_hat.val.copy()
        quat0 = state0[3:7].copy()
        
        # Find Middle Orbital State Once
        mid_os = self.prev_os.average(os)
        CG5 = CG5Coefficients()
        mid_os = [self.prev_os.average(os,CG5.c[j]) for j in range(5)]

        dyn_state0 = self.est_sat.noiseless_rk4(x=state0[0:self.est_sat.state_len], u=u, dt=self.dt, orbital_state0=self.prev_os, orbital_state1=os, mid_orbital_state=mid_os, quat_as_vec=False)        

        which_sensors: List[bool] = [True for j in self.est_sat.attitude_sensors]

        for j, sensor in enumerate(self.est_sat.attitude_sensors):
            if isinstance(sensor, SunSensor):
                if sensor.clean_reading(x=dyn_state0, os=os)<1e-10:
                    which_sensors[j] = False

        sens_vec_len = sum([self.est_sat.sensors[j].output_length for j in range(len(self.est_sat.sensors)) if which_sensors[j]])
        #generate sigma points of augmemted state--state itself, including actuator bias values, disturbance values, sensor bias values; sensor noise , control noise to use, snesor noise to use, possibly integration noise to use.
        L,pts,wts_m,wts_c,sig0 = self.make_pts_and_wts(state0,which_sensors)
        sigma_state_len = len(state0) - 1 + self.quat_as_vec
        post_pts = np.nan*np.ones((2*L+1,sigma_state_len))
        post_sens = np.nan*np.ones((2*L+1,sens_vec_len))
        satj = copy.deepcopy(self.est_sat)
        whichj = which_sensors.copy()

        extra_obj = extra()
        extra_obj.cov0 = self.x_hat.cov.copy()
        extra_obj.sig0 = sig0
        extra_obj.mean0 = sig0[0,:].copy()
        for j in range(2*L+1): #TODO vectorize
            [full_pre_statej,sens_noise_j,control_noise_j,int_noise_extra_j] = pts[j]

            self.sat_match(satj,full_pre_statej)
            post_dyn_state_j = satj.noiseless_rk4(x=full_pre_statej[0:self.est_sat.state_len],u=u + control_noise_j,dt=self.dt,orbital_state0=self.prev_os,orbital_state1=os,mid_orbital_state = mid_os,quat_as_vec = False)

            if j == 0:
                post_quat = post_dyn_state_j[3:7]#can happen before integration noise is added because j=0 has 0 integration noise
            post_statej,post_full_statej = self.new_post_state(full_pre_statej[self.est_sat.state_len:],post_dyn_state_j,int_noise_extra_j,post_quat)
            post_pts[j,:] = post_statej.copy()

            self.sat_match(satj,post_full_statej)
            sensj = satj.sensor_readings(x=post_full_statej[0:self.est_sat.state_len], os=os)
            post_sens[j,:] = sensj.copy()

        state1 = np.dot(wts_m,post_pts)
        dquat1 = vec3_to_quat(state1[3:6],self.vec_mode)
        quat1 = quat_mult(post_quat,dquat1)
        pred_dyn_state = np.concatenate([state1[0:3],quat1,state1[6:self.est_sat.state_len-1],state1[self.est_sat.state_len-1:]])

        sens1 = np.dot(wts_m,post_sens)
        extra_obj.mean1 = pred_dyn_state.copy()

        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1
        cov1 = sum([wts_c[j]*np.outer(pts_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)])

        covyy = sum([wts_c[j]*np.outer(sens_diff[j,:],sens_diff[j,:]) for j in range(2*L+1)],0*np.eye(sens_vec_len))#sum([wts_c[i]*(post_sens[j,:]-sens1)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        covyy += self.est_sat.sensor_cov()
        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))#sum([wts_c[i]*(state_pts_err[i]-x1_cut_red)@(sens_pts[i]-sens1).T for i in range(2*L+1)])

        try:
            Kk = scipy.linalg.solve(covyy,covyx)
        except:
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')


        extra_obj.senscov = covyy.copy()
        # extra_obj.sens_state = state1 +  (sensors[which_sensors]-sens1)@covyx
        extra_obj.sens1 = sens1.copy()
        self.sat_match(satj,pred_dyn_state)
        extra_obj.sens_of_state1 = satj.sensor_readings(x=pred_dyn_state[0:self.est_sat.state_len], os=os)

        extra_obj.sens_sig = post_sens.copy()

        state2 = state1 + (sensors[which_sensors]-sens1)@Kk
        cov2 = cov1 - Kk.T@covyy@Kk
        cov2 = 0.5*(cov2 + cov2.T)

        if not self.quat_as_vec:
            dvec3 = state2[3:6]
            dquat = vec3_to_quat(dvec3,self.vec_mode)
            quat = quat_mult(post_quat,dquat)
            state2 = np.concatenate([state2[0:3],quat,state2[6:self.est_sat.state_len-1],state2[self.est_sat.state_len-1:]])
        else:
            state20 = np.copy(state2)
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T@cov2@norm_jac


        self.sat_match(satj,state2)
        extra_obj.sens_of_state2 = satj.sensor_readings(x=state2[0:self.est_sat.state_len], os=os).copy()

        extra_obj.cov1 = cov1
        extra_obj.cov2 = cov2
        tmp = np.zeros(state2.shape[0])
        tmp[3:7] = post_quat
        extra_obj.sig1 = self.add_to_state(tmp,post_pts).copy()
        extra_obj.mean2 = state2.copy()
        extra_obj.sens_state = self.add_to_state(tmp,state1 +  (sensors[which_sensors]-sens1)@covyx)



        return EstimatedArray(val=state2, cov=cov2)

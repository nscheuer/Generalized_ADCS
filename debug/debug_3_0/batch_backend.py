"""Complete fixed-step array lowerings for the satellite block graph."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .prototype import INERTIA, J2, J3, J4, MASS, MU_EARTH, OMEGA_EARTH, R_EARTH, SUN_ECI


@dataclass
class BatchResult:
    time: np.ndarray
    attitude: np.ndarray
    wall_time: float
    compile_time: float
    backend: str
    device: str
    orbit: np.ndarray | None = None
    estimate: np.ndarray | None = None

    @property
    def satellite_steps_per_second(self) -> float:
        return self.attitude.shape[0] * self.attitude.shape[1] / max(self.wall_time, 1e-15)


@njit(fastmath=True)
def _unit(x):
    return x / max(math.sqrt(np.dot(x, x)), 1e-15)


@njit(fastmath=True)
def _qmat(q):
    q = q / max(math.sqrt(np.dot(q, q)), 1e-15)
    w, x, y, z = q
    out = np.empty((3, 3))
    out[0, 0] = 1-2*(y*y+z*z); out[0, 1] = 2*(x*y-z*w); out[0, 2] = 2*(x*z+y*w)
    out[1, 0] = 2*(x*y+z*w); out[1, 1] = 1-2*(x*x+z*z); out[1, 2] = 2*(y*z-x*w)
    out[2, 0] = 2*(x*z-y*w); out[2, 1] = 2*(y*z+x*w); out[2, 2] = 1-2*(x*x+y*y)
    return out


@njit(fastmath=True)
def _magnetic(r):
    rh = _unit(r); d = np.array([0., 0., 1.])
    return 3.12e-5*(R_EARTH/math.sqrt(np.dot(r, r)))**3*(3*rh*np.dot(d, rh)-d)


@njit(fastmath=True)
def _zonal(r, degree):
    radius = math.sqrt(np.dot(r, r)); h = r/radius; s = h[2]
    radial = -MU_EARTH/radius**2; ds = 0.
    if degree >= 2:
        radial += MU_EARTH*3*J2*R_EARTH**2/radius**4*.5*(3*s*s-1)
        ds -= MU_EARTH*J2*R_EARTH**2/radius**3*(3*s)
    if degree >= 3:
        radial += MU_EARTH*4*J3*R_EARTH**3/radius**5*.5*(5*s**3-3*s)
        ds -= MU_EARTH*J3*R_EARTH**3/radius**4*.5*(15*s*s-3)
    if degree >= 4:
        radial += MU_EARTH*5*J4*R_EARTH**4/radius**6*(35*s**4-30*s*s+3)/8
        ds -= MU_EARTH*J4*R_EARTH**4/radius**5*(140*s**3-60*s)/8
    return radial*h+(ds/radius)*(np.array([0., 0., 1.])-s*h)


@njit(fastmath=True)
def _orbit_rhs(y, force, degree):
    out = np.empty(6); out[:3] = y[3:]; out[3:] = _zonal(y[:3], degree)+force/MASS; return out


@njit(fastmath=True)
def _orbit_step(y, force, dt, degree, order):
    k1 = _orbit_rhs(y, force, degree)
    if order == 2: return y+dt*_orbit_rhs(y+.5*dt*k1, force, degree)
    k2=_orbit_rhs(y+.5*dt*k1,force,degree); k3=_orbit_rhs(y+.5*dt*k2,force,degree); k4=_orbit_rhs(y+dt*k3,force,degree)
    return y+dt*(k1+2*k2+2*k3+k4)/6


@njit(fastmath=True)
def _att_rhs(y, torque, rw):
    w=y[:3]; q=y[3:7]; h=y[7:]; out=np.empty(10)
    out[:3]=np.linalg.solve(INERTIA,torque-np.cross(w,INERTIA@w+h)); out[3]=-.5*np.dot(q[1:],w)
    out[4]=.5*(q[0]*w[0]+q[2]*w[2]-q[3]*w[1]); out[5]=.5*(q[0]*w[1]+q[3]*w[0]-q[1]*w[2]); out[6]=.5*(q[0]*w[2]+q[1]*w[1]-q[2]*w[0]); out[7:]=-rw
    return out


@njit(fastmath=True)
def _att_step(y, torque, rw, dt, order):
    k1=_att_rhs(y,torque,rw)
    if order==2: out=y+dt*_att_rhs(y+.5*dt*k1,torque,rw)
    else:
        k2=_att_rhs(y+.5*dt*k1,torque,rw); k3=_att_rhs(y+.5*dt*k2,torque,rw); k4=_att_rhs(y+dt*k3,torque,rw); out=y+dt*(k1+2*k2+2*k3+k4)/6
    out[3:7]/=max(math.sqrt(np.dot(out[3:7],out[3:7])),1e-15); return out


@njit(fastmath=True)
def _qmul(a,b):
    out=np.empty(4); out[0]=a[0]*b[0]-np.dot(a[1:],b[1:]); out[1]=a[0]*b[1]+b[0]*a[1]+a[2]*b[3]-a[3]*b[2]; out[2]=a[0]*b[2]+b[0]*a[2]+a[3]*b[1]-a[1]*b[3]; out[3]=a[0]*b[3]+b[0]*a[3]+a[1]*b[2]-a[2]*b[1]; return out


@njit(fastmath=True)
def _rotquat(v):
    a=math.sqrt(np.dot(v,v)); out=np.empty(4)
    if a<1e-12: out[0]=1.; out[1:]=.5*v; return out
    out[0]=math.cos(a/2); out[1:]=math.sin(a/2)*v/a; return out


@njit(fastmath=True)
def _disturbance(o,a):
    r=o[:3]; v=o[3:]; rot=_qmat(a[3:7]); rel=v-np.cross(OMEGA_EARTH,r); flow=rot.T@rel; area=np.dot(np.array([.02,.03,.06]),np.abs(_unit(flow))); altitude=math.sqrt(np.dot(r,r))-R_EARTH; density=6e-13*math.exp(-(altitude-500000.)/60000.); force=-.5*density*2.2*area*math.sqrt(np.dot(rel,rel))*rel; drag=np.cross(np.array([.012,-.007,.018]),rot.T@force); rb=rot.T@_unit(r); gg=3*MU_EARTH/(np.dot(r,r)**1.5)*np.cross(rb,INERTIA@rb); return force,drag+gg


@njit(parallel=True, fastmath=True)
def _complete_batch(initial_o, initial_a, gains, gyro_n, mag_n, sun_n, actuator_n, dt, orbit_stride, subsystem_stride, orbit_order, attitude_order, degree, use_sensors):
    runs=initial_a.shape[0]; steps=gyro_n.shape[1]; oo=np.empty((steps,runs,6)); aa=np.empty((steps,runs,10)); ee=np.empty((steps,runs,10))
    for run in prange(runs):
        o=initial_o[run].copy(); a=initial_a[run].copy(); e=np.zeros(10); e[:3]=a[:3]; e[3:7]=a[3:7]; bias=np.zeros(3); cmd=np.zeros(6); force=np.zeros(3); tq=np.zeros(3)
        for step in range(steps):
            if step%subsystem_stride==0:
                force,tq=_disturbance(o,a)
                if use_sensors:
                    rot=_qmat(a[3:7]).T; gyro=a[:3]+bias+gyro_n[run,step]; mag=rot@_magnetic(o[:3])+mag_n[run,step]; sun=_unit(rot@SUN_ECI+sun_n[run,step]); omega=gyro-e[7:]; e[3:7]=_qmul(e[3:7],_rotquat(omega*dt*subsystem_stride)); er=np.cross(_unit(mag),_unit(_qmat(e[3:7]).T@_magnetic(o[:3]))); es=np.cross(_unit(sun),_unit(_qmat(e[3:7]).T@SUN_ECI)); corr=.08*er+.12*es; e[3:7]=_qmul(e[3:7],_rotquat(corr)); e[3:7]/=max(math.sqrt(np.dot(e[3:7],e[3:7])),1e-15); bias-=2e-4*corr; e[:3]=omega; e[7:]=bias
                else:
                    e[:3]=a[:3]; e[3:7]=a[3:7]; e[7:]=0.
                b=_qmat(e[3:7]).T@_magnetic(o[:3]); sign=1. if e[3]>=0 else -1.; cmd[3:]=np.clip(-gains[run]*(.006*sign*e[4:7]+.025*e[:3]),-.002,.002); cmd[:3]=np.clip(.06*np.cross(a[7:10],b)/max(np.dot(b,b),1e-14),-.2,.2)
            if step%orbit_stride==0: o=_orbit_step(o,force,dt*orbit_stride,degree,orbit_order)
            b=_qmat(a[3:7]).T@_magnetic(o[:3]); mtq=cmd[:3]+actuator_n[run,step]; rw=cmd[3:]+actuator_n[run,step]; a=_att_step(a,rw+np.cross(mtq,b),rw,dt,attitude_order); oo[step,run]=o; aa[step,run]=a; ee[step,run]=e
    return oo,aa,ee


def _inputs(runs,steps,seed,initial_orbit_state=None,initial_attitude_state=None):
    rng=np.random.default_rng(seed); o=np.asarray(initial_orbit_state if initial_orbit_state is not None else [7e6,0,0,0,7546,0],float); a=np.asarray(initial_attitude_state if initial_attitude_state is not None else np.r_[np.deg2rad([1.2,-.8,.5]),[.996,.04,-.055,.035],np.zeros(3)],float)
    if o.shape!=(6,) or a.shape!=(10,): raise ValueError("initial orbit/attitude shapes must be (6,) and (10,)")
    io=np.tile(o,(runs,1)); ia=np.tile(a,(runs,1)); ia[:,:3]+=rng.normal(0,np.deg2rad(.15),(runs,3)); ia[:,3:7]/=np.linalg.norm(ia[:,3:7],axis=1,keepdims=True)
    return io,ia,rng.normal(1,.06,runs),rng.normal(0,2e-4,(runs,steps,3)),rng.normal(0,2e-7,(runs,steps,3)),rng.normal(0,2e-3,(runs,steps,3)),rng.normal(0,2e-5,(runs,steps,3))


def _args(runs,duration,dt,seed,initial_attitude_state,initial_orbit_state): return _inputs(runs,int(math.floor(duration/dt+1e-9))+1,seed,initial_orbit_state,initial_attitude_state)


def run_numba_monte_carlo(runs=3,duration=12.,dt=.01,seed=11,*,controller_stride=10,initial_attitude_state=None,initial_orbit_state=None,orbit_stride=1,subsystem_stride=1,orbit_integrator="rk4",attitude_integrator="rk4",orbit_degree=2,sensors=True):
    steps=int(math.floor(duration/dt+1e-9))+1; data=_args(runs,duration,dt,seed,initial_attitude_state,initial_orbit_state); oo=2 if orbit_integrator=="rk2" else 4; aa=2 if attitude_integrator=="rk2" else 4
    started=time.perf_counter(); _complete_batch(data[0][:1],data[1][:1],data[2][:1],data[3][:1,:2],data[4][:1,:2],data[5][:1,:2],data[6][:1,:2],dt,orbit_stride,subsystem_stride,oo,aa,orbit_degree,sensors); compile_time=time.perf_counter()-started
    started=time.perf_counter(); orbit,attitude,estimate=_complete_batch(*data,dt,orbit_stride,subsystem_stride,oo,aa,orbit_degree,sensors); wall=time.perf_counter()-started
    return BatchResult(np.arange(steps)*dt,attitude,wall,compile_time,"numba-fixed-complete","CPU",orbit,estimate)


def run_jax_monte_carlo(runs=3,duration=12.,dt=.01,seed=11,*,require_gpu=True,controller_stride=10,initial_attitude_state=None,initial_orbit_state=None,orbit_stride=1,subsystem_stride=1,orbit_integrator="rk4",attitude_integrator="rk4",orbit_degree=2,sensors=True):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    try:
        import jax; import jax.numpy as jnp; from jax import lax
    except ImportError as exc: raise RuntimeError("JAX is not installed") from exc
    jax.config.update("jax_enable_x64",True)
    try: gpu=jax.devices("gpu")
    except RuntimeError: gpu=[]
    if require_gpu and not gpu: raise RuntimeError("JAX is installed but no GPU device is available")
    device=gpu[0] if gpu else jax.devices("cpu")[0]; steps=int(math.floor(duration/dt+1e-9))+1; data=_args(runs,duration,dt,seed,initial_attitude_state,initial_orbit_state); gains_arr=jnp.asarray(data[2]); I=jnp.asarray(np.diag(INERTIA)); O=jnp.asarray(OMEGA_EARTH); S=jnp.asarray(SUN_ECI); D=jnp.array([0.,0.,1.])
    def unit(x): return x/jnp.maximum(jnp.linalg.norm(x,axis=-1,keepdims=True),1e-15)
    def qmat(q):
        q=unit(q); w,x,y,z=jnp.moveaxis(q,-1,0); return jnp.stack([jnp.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],-1),jnp.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],-1),jnp.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1)],-2)
    def mag(r): rh=unit(r); return 3.12e-5*(R_EARTH/jnp.linalg.norm(r,axis=-1,keepdims=True))**3*(3*rh*jnp.sum(rh*D,axis=-1,keepdims=True)-D)
    def zonal(r):
        rad=jnp.linalg.norm(r,axis=-1,keepdims=True); h=r/rad; s=h[...,2:3]; ar=-MU_EARTH/rad**2; ds=jnp.zeros_like(s)
        if orbit_degree>=2: ar+=MU_EARTH*3*J2*R_EARTH**2/rad**4*.5*(3*s*s-1); ds-=MU_EARTH*J2*R_EARTH**2/rad**3*(3*s)
        if orbit_degree>=3: ar+=MU_EARTH*4*J3*R_EARTH**3/rad**5*.5*(5*s**3-3*s); ds-=MU_EARTH*J3*R_EARTH**3/rad**4*.5*(15*s*s-3)
        if orbit_degree>=4: ar+=MU_EARTH*5*J4*R_EARTH**4/rad**6*(35*s**4-30*s*s+3)/8; ds-=MU_EARTH*J4*R_EARTH**4/rad**5*(140*s**3-60*s)/8
        return ar*h+(ds/rad)*(jnp.array([0.,0.,1.])-s*h)
    def orhs(y,f): return jnp.concatenate([y[...,3:],zonal(y[...,:3])+f/MASS],-1)
    def ostep(y,f,h):
        k1=orhs(y,f)
        if orbit_integrator=="rk2": return y+h*orhs(y+.5*h*k1,f)
        k2=orhs(y+.5*h*k1,f); k3=orhs(y+.5*h*k2,f); k4=orhs(y+h*k3,f); return y+h*(k1+2*k2+2*k3+k4)/6
    def qmul(a,b): return jnp.concatenate([a[...,:1]*b[...,:1]-jnp.sum(a[...,1:]*b[...,1:],-1,keepdims=True),a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+jnp.cross(a[...,1:],b[...,1:])],-1)
    def rq(v): a=jnp.linalg.norm(v,axis=-1,keepdims=True); return jnp.concatenate([jnp.cos(a/2),jnp.where(a<1e-12,.5*v,jnp.sin(a/2)*v/a)],-1)
    def arhs(y,t,rw):
        w=y[...,:3]; q=y[...,3:7]; h=y[...,7:]; wd=(t-jnp.cross(w,I*w+h))/I; pure=jnp.concatenate([jnp.zeros_like(w[...,:1]),w],-1); return jnp.concatenate([wd,.5*qmul(q,pure),-rw],-1)
    def astep(y,t,rw):
        k1=arhs(y,t,rw)
        if attitude_integrator=="rk2": out=y+dt*arhs(y+.5*dt*k1,t,rw)
        else: k2=arhs(y+.5*dt*k1,t,rw); k3=arhs(y+.5*dt*k2,t,rw); k4=arhs(y+dt*k3,t,rw); out=y+dt*(k1+2*k2+2*k3+k4)/6
        return out.at[...,3:7].set(unit(out[...,3:7]))
    def disturbance(o,a):
        r=o[...,:3]; v=o[...,3:]; rot=jnp.swapaxes(qmat(a[...,3:7]),-1,-2); rel=v-jnp.cross(O,r); flow=jnp.einsum('...ij,...j->...i',rot,rel); area=jnp.sum(jnp.array([.02,.03,.06])*jnp.abs(unit(flow)),-1); den=6e-13*jnp.exp(-(jnp.linalg.norm(r,axis=-1)-R_EARTH-500000.)/60000.); force=-.5*den[...,None]*2.2*area[...,None]*jnp.linalg.norm(rel,axis=-1,keepdims=True)*rel; drag=jnp.cross(jnp.array([.012,-.007,.018]),jnp.einsum('...ij,...j->...i',rot,force)); rb=jnp.einsum('...ij,...j->...i',rot,unit(r)); gg=3*MU_EARTH/jnp.linalg.norm(r,axis=-1,keepdims=True)**3*jnp.cross(rb,rb*I); return force,drag+gg
    def scan(c,item):
        o,a,e,bias,cmd=c; step,gnoise,mnoise,snoise,anoise=item; nf,nt=disturbance(o,a); sub=step%subsystem_stride==0
        def update(x):
            oo,aa,ee,bb,cc=x
            if sensors:
                rot=jnp.swapaxes(qmat(aa[...,3:7]),-1,-2); gyro=aa[...,:3]+bb+gnoise
                m=jnp.einsum('...ij,...j->...i',rot,mag(oo[...,:3]))+mnoise; sv=jnp.einsum('...ij,...j->...i',rot,S)+snoise
                omega=gyro-ee[...,7:]; eq=qmul(ee[...,3:7],rq(omega*(dt*subsystem_stride)))
                predicted_m=jnp.einsum('...ij,...j->...i',jnp.swapaxes(qmat(eq),-1,-2),mag(oo[...,:3]))
                predicted_s=jnp.einsum('...ij,...j->...i',jnp.swapaxes(qmat(eq),-1,-2),S)
                er=jnp.cross(unit(m),unit(predicted_m)); es=jnp.cross(unit(sv),unit(predicted_s)); corr=.08*er+.12*es
                eq=unit(qmul(eq,rq(corr))); ee=jnp.concatenate([omega,eq,bb-2e-4*corr],-1)
            else: ee=jnp.concatenate([aa[...,:3],aa[...,3:7],jnp.zeros_like(aa[...,:3])],-1)
            return oo,aa,ee,bb,cc
        o,a,e,bias,cmd=lax.cond(sub,update,lambda x:x,(o,a,e,bias,cmd)); b=jnp.einsum('...ij,...j->...i',jnp.swapaxes(qmat(e[...,3:7]),-1,-2),mag(o[...,:3])); sign=jnp.where(e[...,3:4]>=0,1.,-1.); rw=jnp.clip(-gains_arr[:,None]*(.006*sign*e[...,4:7]+.025*e[...,:3]),-.002,.002); mtq=jnp.clip(.06*jnp.cross(a[...,7:10],b)/jnp.maximum(jnp.sum(b*b,-1,keepdims=True),1e-14),-.2,.2); an=jnp.concatenate([mtq+anoise,rw+anoise],-1); torque=an[...,3:]+jnp.cross(an[...,:3],jnp.einsum('...ij,...j->...i',jnp.swapaxes(qmat(a[...,3:7]),-1,-2),mag(o[...,:3]))); o=lax.cond(step%orbit_stride==0,lambda x:ostep(x[0],x[1],dt*orbit_stride),lambda x:x[0],(o,nf)); a=astep(a,torque,an[...,3:]); return (o,a,e,bias,cmd),(o,a,e)
    def kernel(io,ia,g,gn,mn,sn,an):
        carry=(io,ia,jnp.zeros((runs,10)),jnp.zeros((runs,3)),jnp.zeros((runs,6))); items=(jnp.arange(steps),jnp.swapaxes(gn,0,1),jnp.swapaxes(mn,0,1),jnp.swapaxes(sn,0,1),jnp.swapaxes(an,0,1)); return lax.scan(scan,carry,items)[1]
    arrays=[jax.device_put(jnp.asarray(x),device) for x in data]; compiled=jax.jit(kernel); started=time.perf_counter(); first=compiled(*arrays); [x.block_until_ready() for x in first]; compile_time=time.perf_counter()-started; started=time.perf_counter(); out=compiled(*arrays); [x.block_until_ready() for x in out]; wall=time.perf_counter()-started; return BatchResult(np.arange(steps)*dt,np.asarray(out[1]),wall,compile_time,"jax-fixed-complete",str(device),np.asarray(out[0]),np.asarray(out[2]))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)


def create_saltro_satellite(sat: Satellite, saltro_py) -> object:
    cpp_sat = saltro_py.Satellite()
    cpp_sat.setInertia(sat.J_COM)
    
    for mtq in sat.mtq_actuators:
        cpp_sat.addMTQ(mtq.axis, mtq.u_max)
    
    for rw in sat.rw_actuators:
        cpp_sat.addRW(rw.axis, rw.u_max, rw.J, rw.h, rw.h_max)
    
    return cpp_sat


def plot_trajectory_summary(
    X: np.ndarray,
    U: np.ndarray,
    dt: float,
    out_path: str,
) -> None:
    """Create and save a compact trajectory summary plot."""
    t_x = np.arange(X.shape[1]) * dt
    t_u = np.arange(U.shape[1]) * dt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    # Angular velocity
    axes[0].plot(t_x, X[0, :], label="wx")
    axes[0].plot(t_x, X[1, :], label="wy")
    axes[0].plot(t_x, X[2, :], label="wz")
    axes[0].set_title("Angular Velocity")
    axes[0].set_ylabel("rad/s")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Quaternion
    axes[1].plot(t_x, X[3, :], label="q0")
    axes[1].plot(t_x, X[4, :], label="q1")
    axes[1].plot(t_x, X[5, :], label="q2")
    axes[1].plot(t_x, X[6, :], label="q3")
    axes[1].set_title("Quaternion")
    axes[1].set_ylabel("-")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # RW momentum
    if X.shape[0] > 7:
        for i in range(7, X.shape[0]):
            axes[2].plot(t_x, X[i, :], label=f"h{i-7}")
        axes[2].legend(loc="upper right")
    axes[2].set_title("Reaction Wheel Momentum")
    axes[2].set_ylabel("Nms")
    axes[2].grid(True, alpha=0.3)

    # Controls
    for i in range(U.shape[0]):
        axes[3].plot(t_u, U[i, :], label=f"u{i}")
    axes[3].set_title("Control Inputs")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("command")
    axes[3].legend(loc="upper right", ncol=2)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    import saltro_py

    np.random.seed(1)
    t0 = 0.0
    tf = 1000.0
    dt = 10.0
    N = int(np.ceil((tf - t0) / dt)) + 1

    mtm_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]
    rw_max_torque = 7*0.001
    rw_J = 0.001
    rw_h0 = 5*0.001
    rw_hmax = 16.2*0.001
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    acts = mtqs+rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    w0 = np.array([0.01, 0, 0])
    q0 = random_n_unit_vec(4)
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0, rw_h0, rw_h0])
    x = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    t_start = 0.22
    R = np.array([7000.0, 0.0, 0.0])
    V = np.array([0.0, 8.0, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=t_start, R=R, V=V)
    
    # Planner Settings
    planner_settings = PlannerSettings(est_sat=real_sat)
    print("Created Planner Settings")
    
    # Convert to C++ object
    cpp_settings = planner_settings.to_cpp()
    print("Created C++ Planner Settings")
    
    # Convert Satellite to C++ object
    cpp_satellite = create_saltro_satellite(real_sat, saltro_py)
    print("Created C++ Satellite")

    jtime = np.ascontiguousarray(
        t_start + np.arange(N, dtype=np.float64) * dt * TimeConstants.sec2cent,
        dtype=np.float64,
    )

    # One-shot 90deg slew reference around body-Y axis, held constant in time.
    q_ref = np.array([np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0], dtype=np.float64)
    q_goal = np.tile(q_ref.reshape(4, 1), (1, N))

    body_boresight = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    boresight = np.tile(body_boresight.reshape(3, 1), (1, N))

    # ADCS orbit states are in km / km/s. Convert to SI for SALTRO.
    r0_m = np.asarray(os0.R, dtype=np.float64) * 1.0e3
    v0_mps = np.asarray(os0.V, dtype=np.float64) * 1.0e3

    try:
        ok, X, U, K = saltro_py.trajOpt(
            cpp_settings,
            cpp_satellite,
            np.asarray(x, dtype=np.float64),
            r0_m,
            v0_mps,
            jtime,
            q_goal,
            boresight,
        )
    except RuntimeError as err:
        msg = str(err)
        if "apogee altitude above LEO bounds" not in msg:
            raise

        # Keep requested ADCS defaults (|R|=7000 km, |V|=8 km/s) as primary case,
        # but allow a nearby valid speed for SALTRO's strict LEO validator.
        v0_retry = np.asarray([0.0, 7.5e3, 0.0], dtype=np.float64)
        print("SALTRO rejected V=8 km/s for LEO bounds; retrying with V=7.5 km/s for debug output.")
        ok, X, U, K = saltro_py.trajOpt(
            cpp_settings,
            cpp_satellite,
            np.asarray(x, dtype=np.float64),
            r0_m,
            v0_retry,
            jtime,
            q_goal,
            boresight,
        )

    if not ok:
        raise RuntimeError("SALTRO trajOpt failed")

    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    n_out = X.shape[1]
    n_u = U.shape[0]
    n_red = int(cpp_satellite.reducedStateDim)

    assert X.shape[0] == int(cpp_satellite.stateDim), f"Unexpected X shape: {X.shape}"
    assert U.shape[1] == n_out, f"U time dimension mismatch: U={U.shape}, X={X.shape}"
    assert K.shape == (n_u, n_red * n_out), (
        f"Unexpected K shape {K.shape}; expected ({n_u}, {n_red * n_out})"
    )

    print("SALTRO trajOpt succeeded")
    print(f"N request={N}, N out={n_out}")
    print(f"X shape={X.shape}, U shape={U.shape}, K shape={K.shape}")

    out_plot = os.path.join(current_dir, "output", "saltro_trajectory_summary.png")
    plot_trajectory_summary(X=X, U=U, dt=dt, out_path=out_plot)
    print(f"Saved trajectory plot: {out_plot}")
    




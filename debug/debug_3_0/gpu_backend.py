"""Optional JAX batch backend for the block-graph prototype.

The event queue remains a CPU concern.  This backend lowers the regular,
minimum-tick subset of the same model configuration to one batched JAX scan.
It deliberately requires a real GPU by default so a CPU fallback is never
misreported as GPU execution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .prototype import J2, J3, J4, MASS, MU_EARTH, R_EARTH, ModelConfig


@dataclass
class GPUBatchResult:
    time: np.ndarray
    attitude: np.ndarray  # (time, runs, 10): w, q, h
    estimated_q: np.ndarray  # (time, runs, 4)
    altitude: np.ndarray  # (time, runs)
    seeds: tuple[int, ...]
    wall_time: float
    device: str


def _lattice(config: ModelConfig) -> tuple[float, dict[str, tuple[int, int]]]:
    base = min(t.period for t in config.timings.values())
    lattice: dict[str, tuple[int, int]] = {}
    for name, timing in config.timings.items():
        stride_f = timing.period / base
        phase_f = timing.phase / base
        stride, phase = int(round(stride_f)), int(round(phase_f))
        if abs(stride_f - stride) > 1e-9 or abs(phase_f - phase) > 1e-9:
            raise ValueError(
                f"GPU backend requires period and phase on the {base:g}s lattice; "
                f"block {name!r} has period={timing.period:g}, phase={timing.phase:g}"
            )
        lattice[name] = (stride, phase)
    return base, lattice


def run_gpu_monte_carlo(
    config: ModelConfig,
    seeds: Sequence[int] = (101, 102, 103),
    *,
    require_gpu: bool = True,
) -> GPUBatchResult:
    """Run a regular block schedule as one JAX batch.

    Set ``require_gpu=False`` only for development/CI validation.  The returned
    device field always states where JAX actually executed the model.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax, random
    except ImportError as exc:
        raise RuntimeError(
            "The GPU demo needs the optional JAX dependency. Install the JAX build "
            "appropriate for the machine's CUDA/ROCm platform."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "JAX is installed, but no GPU device is available. The demo did not run "
            "and will not label a CPU fallback as GPU execution."
        )
    device = gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]
    base_dt, lattice = _lattice(config)
    count = len(seeds)
    steps = int(math.floor(config.duration / base_dt + 1e-9)) + 1
    inertia = jnp.array([0.035, 0.045, 0.052])
    inv_inertia = 1.0 / inertia
    sun_eci = jnp.array([0.92541658, 0.33682409, 0.17364818])
    earth_omega = jnp.array([0.0, 0.0, 7.2921150e-5])

    def unit(x):
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-15)

    def qmul(a, b):
        av, bv = a[..., 1:], b[..., 1:]
        return jnp.concatenate(
            [(a[..., :1] * b[..., :1] - jnp.sum(av * bv, axis=-1, keepdims=True)),
             a[..., :1] * bv + b[..., :1] * av + jnp.cross(av, bv)], axis=-1
        )

    def rotvec_q(v):
        angle = jnp.linalg.norm(v, axis=-1, keepdims=True)
        scale = jnp.where(angle < 1e-10, 0.5 - angle**2 / 48.0, jnp.sin(angle / 2) / angle)
        return unit(jnp.concatenate([jnp.cos(angle / 2), scale * v], axis=-1))

    def qmat(q):
        q = unit(q)
        w, x, y, z = [q[..., i] for i in range(4)]
        row0 = jnp.stack([1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], axis=-1)
        row1 = jnp.stack([2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], axis=-1)
        row2 = jnp.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)], axis=-1)
        return jnp.stack([row0, row1, row2], axis=-2)

    def mat_t_vec(m, v):
        return jnp.einsum("...ji,...j->...i", m, v)

    def zonal(r):
        radius = jnp.linalg.norm(r, axis=-1, keepdims=True)
        rhat = r / radius
        s = rhat[..., 2:3]
        radial = -MU_EARTH / radius**2
        ds = jnp.zeros_like(radial)
        p2, dp2 = 0.5 * (3*s*s - 1), 3*s
        scale2 = J2 * R_EARTH**2 / radius**4
        radial += MU_EARTH * 3 * scale2 * p2
        ds -= MU_EARTH * J2 * R_EARTH**2 / radius**3 * dp2
        if config.orbit_degree >= 4:
            p3, dp3 = 0.5 * (5*s**3 - 3*s), 0.5 * (15*s*s - 3)
            p4, dp4 = (35*s**4 - 30*s*s + 3)/8, (140*s**3 - 60*s)/8
            radial += MU_EARTH * 4 * J3 * R_EARTH**3 / radius**5 * p3
            radial += MU_EARTH * 5 * J4 * R_EARTH**4 / radius**6 * p4
            ds -= MU_EARTH * J3 * R_EARTH**3 / radius**4 * dp3
            ds -= MU_EARTH * J4 * R_EARTH**4 / radius**5 * dp4
        zaxis = jnp.array([0.0, 0.0, 1.0])
        return radial * rhat + (ds / radius) * (zaxis - s * rhat)

    def orbit_rhs(y, force):
        return jnp.concatenate([y[..., 3:], zonal(y[..., :3]) + force / MASS], axis=-1)

    def orbit_step(y, force, dt):
        k1 = orbit_rhs(y, force)
        k2 = orbit_rhs(y + 0.5*dt*k1, force)
        if config.orbit_integrator == "rk2":
            return y + dt*k2
        k3 = orbit_rhs(y + 0.5*dt*k2, force)
        k4 = orbit_rhs(y + dt*k3, force)
        return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def attitude_rhs(y, torque, rw_torque):
        w, q, h = y[..., :3], y[..., 3:7], y[..., 7:]
        wdot = (torque - jnp.cross(w, inertia*w + h)) * inv_inertia
        pure_w = jnp.concatenate([jnp.zeros_like(w[..., :1]), w], axis=-1)
        qdot = 0.5 * qmul(q, pure_w)
        return jnp.concatenate([wdot, qdot, -rw_torque], axis=-1)

    def attitude_step(y, torque, rw_torque, dt):
        k1 = attitude_rhs(y, torque, rw_torque)
        k2 = attitude_rhs(y + 0.5*dt*k1, torque, rw_torque)
        if config.attitude_integrator == "rk2":
            out = y + dt*k2
        else:
            k3 = attitude_rhs(y + 0.5*dt*k2, torque, rw_torque)
            k4 = attitude_rhs(y + dt*k3, torque, rw_torque)
            out = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        return out.at[..., 3:7].set(unit(out[..., 3:7]))

    def mag_eci(r):
        rh = unit(r)
        dip = jnp.array([0.0, 0.0, 1.0])
        scale = 3.12e-5 * (R_EARTH / jnp.linalg.norm(r, axis=-1, keepdims=True))**3
        return scale * (3*rh*jnp.sum(dip*rh, axis=-1, keepdims=True) - dip)

    def due(step, name):
        stride, phase = lattice[name]
        return (step >= phase) & (((step - phase) % stride) == 0)

    radius = R_EARTH + 500_000.0
    speed = math.sqrt(MU_EARTH / radius)
    inc = math.radians(51.6)
    orbit0 = np.tile(np.array([radius, 0, 0, 0, speed*math.cos(inc), speed*math.sin(inc)]), (count, 1))
    attitude0 = np.tile(np.r_[np.deg2rad([1.2, -0.8, 0.5]), [0.996, 0.04, -0.055, 0.035], np.zeros(3)], (count, 1))
    attitude0[:, 3:7] /= np.linalg.norm(attitude0[:, 3:7], axis=1, keepdims=True)
    keys = jnp.stack([random.PRNGKey(int(seed)) for seed in seeds])
    z3 = jnp.zeros((count, 3), dtype=jnp.float64)
    q0 = jnp.asarray(attitude0[:, 3:7])
    initial = {
        "orbit": jnp.asarray(orbit0), "attitude": jnp.asarray(attitude0),
        "force": z3, "dist_torque": z3, "gyro": jnp.asarray(attitude0[:, :3]),
        "mag": mat_t_vec(qmat(q0), mag_eci(jnp.asarray(orbit0[:, :3]))),
        "sun": mat_t_vec(qmat(q0), jnp.broadcast_to(sun_eci, (count, 3))),
        "qhat": q0, "bias": z3, "mtq": z3, "rw_cmd": z3,
        "act_torque": z3, "rw_real": z3, "keys": keys,
    }

    def select(flag, new, old):
        return jnp.where(flag, new, old)

    def scan_step(c, step):
        # Orbit and attitude are zero-order-held between their own clock events.
        orb_new = orbit_step(c["orbit"], c["force"], config.timings["orbit"].period)
        c["orbit"] = select(due(step, "orbit"), orb_new, c["orbit"])
        att_new = attitude_step(c["attitude"], c["act_torque"] + c["dist_torque"], c["rw_real"], config.timings["attitude"].period)
        c["attitude"] = select(due(step, "attitude"), att_new, c["attitude"])

        r, v, q = c["orbit"][..., :3], c["orbit"][..., 3:], c["attitude"][..., 3:7]
        rotation = qmat(q)
        relative_v = v - jnp.cross(earth_omega, r)
        flow_body = mat_t_vec(rotation, relative_v)
        area = jnp.sum(jnp.array([0.02, 0.03, 0.06]) * jnp.abs(unit(flow_body)), axis=-1, keepdims=True)
        altitude = jnp.linalg.norm(r, axis=-1, keepdims=True) - R_EARTH
        rho = 6e-13 * jnp.exp(-(altitude - 500_000.0)/60_000.0)
        force = -0.5*rho*2.2*area*jnp.linalg.norm(relative_v, axis=-1, keepdims=True)*relative_v
        drag_body = mat_t_vec(rotation, force)
        drag_torque = jnp.cross(jnp.array([0.012, -0.007, 0.018]), drag_body)
        rbody = mat_t_vec(rotation, unit(r))
        gg = 3*MU_EARTH/jnp.linalg.norm(r, axis=-1, keepdims=True)**3 * jnp.cross(rbody, inertia*rbody)
        c["force"] = select(due(step, "disturbances"), force, c["force"])
        c["dist_torque"] = select(due(step, "disturbances"), drag_torque + gg, c["dist_torque"])

        split = jax.vmap(lambda k: random.split(k, 4))(c["keys"])
        c["keys"] = split[:, 0]
        gyro = c["attitude"][..., :3] + jnp.array([2e-4, -1e-4, 1.5e-4]) + jax.vmap(lambda k: random.normal(k, (3,)))(split[:, 1])*config.gyro_noise_std
        mag = mat_t_vec(rotation, mag_eci(r)) + jax.vmap(lambda k: random.normal(k, (3,)))(split[:, 2])*config.magnetometer_noise_std
        sun = unit(mat_t_vec(rotation, jnp.broadcast_to(sun_eci, (count, 3))) + jax.vmap(lambda k: random.normal(k, (3,)))(split[:, 3])*config.sun_noise_std)
        c["gyro"] = select(due(step, "sensors"), gyro, c["gyro"])
        c["mag"] = select(due(step, "sensors"), mag, c["mag"])
        c["sun"] = select(due(step, "sensors"), sun, c["sun"])

        omega_hat = c["gyro"] - c["bias"]
        qpred = unit(qmul(c["qhat"], rotvec_q(omega_hat*config.timings["estimator"].period)))
        pred_rot = qmat(qpred)
        pred_mag = unit(mat_t_vec(pred_rot, mag_eci(r)))
        pred_sun = unit(mat_t_vec(pred_rot, jnp.broadcast_to(sun_eci, (count, 3))))
        correction = 0.08*jnp.cross(unit(c["mag"]), pred_mag) + 0.12*jnp.cross(unit(c["sun"]), pred_sun)
        qhat_new = unit(qmul(qpred, rotvec_q(correction)))
        bias_new = c["bias"] - 2e-4*correction
        c["qhat"] = select(due(step, "estimator"), qhat_new, c["qhat"])
        c["bias"] = select(due(step, "estimator"), bias_new, c["bias"])

        sign = jnp.where(c["qhat"][..., :1] < 0, -1.0, 1.0)
        rw = jnp.clip(-0.006*sign*c["qhat"][..., 1:] - 0.025*(c["gyro"] - c["bias"]), -2e-3, 2e-3)
        bbody_hat = mat_t_vec(qmat(c["qhat"]), mag_eci(r))
        h = c["attitude"][..., 7:]
        mtq = jnp.clip(0.06*jnp.cross(h, bbody_hat)/jnp.maximum(jnp.sum(bbody_hat*bbody_hat, axis=-1, keepdims=True), 1e-14), -0.2, 0.2)
        c["rw_cmd"] = select(due(step, "controller"), rw, c["rw_cmd"])
        c["mtq"] = select(due(step, "controller"), mtq, c["mtq"])

        rw_real = c["rw_cmd"] + jax.vmap(lambda k: random.normal(k, (3,)))(split[:, 1])*config.rw_noise_std
        mtq_real = c["mtq"] + jax.vmap(lambda k: random.normal(k, (3,)))(split[:, 2])*config.mtq_noise_std
        act_torque = jnp.cross(mtq_real, mat_t_vec(rotation, mag_eci(r))) + rw_real
        c["rw_real"] = select(due(step, "actuators"), rw_real, c["rw_real"])
        c["act_torque"] = select(due(step, "actuators"), act_torque, c["act_torque"])
        output = (c["attitude"], c["qhat"], altitude[..., 0])
        return c, output

    compiled = jax.jit(lambda state: lax.scan(scan_step, state, jnp.arange(steps)))
    started = time.perf_counter()
    with jax.default_device(device):
        _, output = compiled(initial)
        # JAX dispatch is asynchronous; this synchronizes timing and transfer.
        attitude, estimated_q, altitude = [np.asarray(x.block_until_ready()) for x in output]
    elapsed = time.perf_counter() - started
    return GPUBatchResult(
        time=np.arange(steps)*base_dt,
        attitude=attitude,
        estimated_q=estimated_q,
        altitude=altitude,
        seeds=tuple(int(x) for x in seeds),
        wall_time=elapsed,
        device=str(device),
    )

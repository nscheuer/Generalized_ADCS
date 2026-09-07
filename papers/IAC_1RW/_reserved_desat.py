"""Reserved-desaturation allocator -- the LP2 ghost, minimally.

The desat trace showed diverged trials commanding despin 44% of the time while LP-saturated
99.7% of the time: the channel had intent and no budget. This allocator pays the despin
channel FIRST -- a guaranteed wheel torque toward h_target plus the magnetorquer dipole that
cancels its reaction, reserved off the top of the box -- and the pointing request is then
allocated by the stock LP against the REMAINING magnetorquer authority.

This is the minimal form of genACS's open "priority-restoring allocation" question: not the
full two-stage LP (which lives uncommitted in a working tree), but the reservation that
decides whether priority ordering rescues the dump-starved regime at all.
"""
from __future__ import annotations
import numpy as np
from ADCS.satellite_hardware.actuators import MTQ, RW
from papers.IAC_1RW._feedforward import FeedforwardLP


class ReservedDesatLP(FeedforwardLP):
    def __init__(self, *args, desat_tau_frac: float = 0.25,
                 desat_deadband_frac: float = 0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.desat_tau_frac = float(desat_tau_frac)      # of tau_w, guaranteed despin rate
        self.desat_deadband_frac = float(desat_deadband_frac)

    def find_u(self, x_hat, sens, est_sat, os_hat, goal, *args, **kwargs):
        u = np.asarray(super().find_u(x_hat, sens, est_sat, os_hat, goal,
                                      *args, **kwargs), float)
        rw_idx = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
        mtq_idx = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        if not rw_idx or not mtq_idx:
            return u
        xh = np.ravel(np.asarray(x_hat, float))
        B = np.ravel(np.asarray(os_hat.B, float))[:3]
        Bn2 = float(B @ B)
        if Bn2 < 1e-24:
            return u

        i_rw = rw_idx[0]
        rw = est_sat.actuators[i_rw]
        a_hat = np.ravel(rw.axis)[:3]
        tau_w = float(np.ravel(rw.u_max)[0])
        h_max = float(np.ravel(rw.h_max)[0])
        h = float(xh[7]) if xh.size >= 8 else 0.0
        h_tgt = float(self.h_target @ a_hat)
        dh = h - h_tgt
        if abs(dh) < self.desat_deadband_frac * h_max:
            return u

        # Guaranteed despin: hdot = -u, so driving h toward target needs u_rw with the SAME
        # sign as dh. Reaction on the body is -u_des * a_hat; the reserved MTQ dipole cancels
        # the perpendicular part exactly (minimum-norm); the along-B part is the priced
        # pointing residual the reservation deliberately accepts.
        u_des = np.sign(dh) * self.desat_tau_frac * tau_w
        tau_needed = u_des * a_hat                       # body torque to cancel: -(-u a) = +u a
        m_res = np.cross(B, tau_needed) / Bn2            # min-norm dipole for the perp part

        # Reservation: desat's dipole comes off the top; pointing keeps the remainder.
        m_lim = np.array([float(np.ravel(est_sat.actuators[i].u_max)[0]) for i in mtq_idx])
        axes = np.column_stack([np.ravel(est_sat.actuators[i].axis)[:3] for i in mtq_idx])
        m_des_cmd = np.linalg.pinv(axes) @ m_res
        m_des_cmd = np.clip(m_des_cmd, -m_lim, m_lim)
        room = m_lim - np.abs(m_des_cmd)                 # what pointing may still use
        for j, i in enumerate(mtq_idx):
            u[i] = float(np.clip(u[i], -room[j], room[j]) + m_des_cmd[j])
        u[i_rw] = u_des
        return u

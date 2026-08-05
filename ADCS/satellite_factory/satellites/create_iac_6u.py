"""IAC-26 "One Wheel Is Enough" reference bus (paper IAC-26,B4,6A,2,x109468).

**One bus, defined once.** Every campaign in the IAC study (R, A, B, C, D, E, F) draws its
spacecraft from :func:`create_iac_6u_bus`. The companion papers (SSC26-P2-54, SSC26-FT-34)
used different inertias, wheels and control rates and are deliberately *not* reused here.

Reference values (campaign spec §1):

======================  ===========================================================
Quantity                Value
======================  ===========================================================
Form factor             6U, 100 x 200 x 300 mm (z = the 300 mm axis)
Mass                    12 kg
Inertia (about COM)     ``diag(0.13, 0.10, 0.05)`` kg m^2 -- uniform-box estimate
Boresight               body ``+z`` -- the *minor* axis
Magnetorquers           3, principal body axes, ``m_max = 0.2`` A m^2
Reaction wheel (1RW)    axis ``+z`` (boresight-aligned), 2.0 mN m, 15 mN m s
Residual dipole         0.05 A m^2 along ``[1,1,1]/sqrt(3)`` (0.1 = sensitivity case)
cp-cg offset            2 cm on ``+x``
======================  ===========================================================

Three details in here are load-bearing and easy to get wrong:

1. **The cp-cg offset is not optional.** For a uniform box every face satisfies
   ``A_i * d_i = V/2``, so the faceted drag sum ``w = sum_i A_i C_D (n_i . V)_+ r_i`` is
   *parallel to* ``V`` and the kernel's ``w x V`` vanishes identically. With ``COM`` at the
   geometric centre the drag torque is **exactly zero at every attitude** (measured 1.8e-22
   N m), and the same cancellation applies to SRP. The 2 cm offset is the entire source of
   both torques.

2. **The inertia is specified about the COM, and ``Satellite`` wants it about the reference
   origin.** :meth:`Satellite.update_J` applies the parallel-axis theorem
   ``J_COM = J_0 - m(|r|^2 I - r r^T)``, so this factory back-solves ``J_0`` to make the
   *dynamics* see exactly the paper's ``diag(0.13, 0.10, 0.05)``.

3. **``authority_scale`` scales the wheel's momentum limit too.** Campaign E's x-axis is a
   disturbance-to-authority ratio, and the storage side of the saturation condition is
   ``tau_sec * T_orbit <= h_max`` -- which contains neither ``m_max`` nor ``tau_w``. Scaling
   only the torques would leave the saturation boundary stationary while the sweep ran. One
   scalar moves all three and holds the wheel time constant ``h_max/tau_w = 7.5`` s fixed.
"""

from __future__ import annotations

__all__ = ["create_iac_6u_bus", "IAC_6U", "iac_6u_geometry_faces"]

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import (
    Dipole_Disturbance,
    Drag_Disturbance,
    GeometryConfig,
    GeometryFace,
    GG_Disturbance,
    SRP_Disturbance,
)
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_factory.sensors import (
    create_Clydespace_3U_array,
    create_iac_gyro,
    create_iac_magnetometer,
    create_iac_star_tracker,
)
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs
_DEG = np.pi / 180.0


@dataclass(frozen=True)
class _IAC6U:
    """Frozen reference constants for the IAC 6U bus. Change here, nowhere else."""

    #: total mass [kg]
    mass: float = 12.0
    #: body dimensions along +x, +y, +z [m] -- z is the 300 mm (long) axis
    dims_m: tuple = (0.1, 0.2, 0.3)
    #: inertia **about the centre of mass** [kg m^2] (uniform box)
    J_com: tuple = (0.13, 0.10, 0.05)
    #: body boresight -- +z, the minor axis
    boresight: tuple = (0.0, 0.0, 1.0)

    #: per-axis magnetorquer dipole limit [A m^2]
    m_max: float = 0.2
    #: reaction-wheel torque limit [N m]
    tau_w: float = 2.0e-3
    #: reaction-wheel momentum limit [N m s]
    h_max: float = 15.0e-3
    #: rotor inertia [kg m^2]
    J_rw: float = 1.0e-5

    #: residual dipole magnitude [A m^2] (0.1 is the labelled sensitivity case)
    m_res: float = 0.05
    #: residual-dipole direction (unnormalised)
    m_res_dir: tuple = (1.0, 1.0, 1.0)
    #: cp-cg offset [m]
    com_offset_m: float = 0.02
    #: cp-cg offset direction -- **along the boresight / long axis**, and that choice
    #: matters. Drag torque is ``prop. (c_hat x V_hat)``, so an offset parallel to the ram
    #: direction produces **no torque at all**. A ram-locked 6U flies +z nadir and +x along
    #: velocity, so an offset along +x would be exactly the degenerate case and would
    #: silently zero the drag torque for the whole nadir-locked campaign. +z is also the
    #: physically likely axis: a 6U's mass sits unevenly along its long axis (payload at
    #: one end), and it is perpendicular to ram in the ram-locked profile, which is what
    #: makes the secular accumulation in Section IV-A appear at all.
    com_offset_dir: tuple = (0.0, 0.0, 1.0)

    #: per-face drag coefficient
    CD: float = 2.2
    #: specular / diffuse / absorbed optical coefficients (sum to 1)
    eta_s: float = 0.5
    eta_d: float = 0.2
    eta_a: float = 0.3

    @property
    def wheel_tau_s(self) -> float:
        """Wheel time constant ``h_max / tau_w`` [s]."""
        return self.h_max / self.tau_w


IAC_6U = _IAC6U()


def iac_6u_geometry_faces(
    dims_m: Sequence[float] = IAC_6U.dims_m,
    CD: float = IAC_6U.CD,
    eta_s: float = IAC_6U.eta_s,
    eta_d: float = IAC_6U.eta_d,
    eta_a: float = IAC_6U.eta_a,
) -> List[GeometryFace]:
    """Six box faces, centroids about the **geometric** centre (not the COM).

    The drag/SRP kernels take the lever arm as ``centroid - COM``, so keeping the geometry
    on the geometric centre and putting the offset in ``COM`` is what produces the cp-cg
    lever. See the module docstring, point 1.
    """
    dx, dy, dz = (float(d) for d in dims_m)
    faces: List[GeometryFace] = []
    for axis, (extent, area) in enumerate(
        ((dx, dy * dz), (dy, dx * dz), (dz, dx * dy))
    ):
        n = _UV[axis]
        for sign in (+1.0, -1.0):
            faces.append(
                GeometryFace(
                    area=area,
                    centroid=sign * n * extent / 2.0,
                    normal=sign * n,
                    eta_s=eta_s,
                    eta_d=eta_d,
                    eta_a=eta_a,
                    CD=CD,
                )
            )
    return faces


def _J0_from_Jcom(J_com: np.ndarray, mass: float, com: np.ndarray) -> np.ndarray:
    r"""Invert the parallel-axis shift ``Satellite.update_J`` applies.

    ``update_J`` computes :math:`J_{COM} = J_0 - m(\|r\|^2 I - r r^\top)`, so to make the
    dynamics see the paper's :math:`J_{COM}` we hand it
    :math:`J_0 = J_{COM} + m(\|r\|^2 I - r r^\top)`.
    """
    r = np.asarray(com, dtype=float).reshape(3)
    shift = mass * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return np.asarray(J_com, dtype=float) + shift


def create_iac_6u_bus(
    n_rw: int = 1,
    *,
    wheel_axes: Optional[Sequence[np.ndarray]] = None,
    authority_scale: float = 1.0,
    m_max: Optional[float] = None,
    tau_w: Optional[float] = None,
    h_max: Optional[float] = None,
    m_res: Optional[float] = None,
    com_offset_m: Optional[float] = None,
    disturbances: Sequence[str] = ("gg", "drag", "srp", "dipole"),
    estimate_dipole: bool = True,
    sensors: Sequence[str] = ("mtm", "gyro", "star_tracker"),
    estimated: bool = False,
    J_com: Optional[np.ndarray] = None,
    mass: Optional[float] = None,
):
    """Build the IAC 6U reference bus.

    :param n_rw: Number of reaction wheels: 0, 1 or 3. ``1`` mounts the wheel on the
        boresight ``+z``; ``3`` mounts ``+x, +y, +z``. Override with ``wheel_axes``.
    :param wheel_axes: Explicit body-frame wheel axes (Campaign D mounts the single wheel
        boresight / orbit-normal / 45-degrees-between). Overrides the ``n_rw`` default.
    :param authority_scale: Campaign E's single actuator-authority knob ``s``. Scales
        ``m_max``, ``tau_w`` **and** ``h_max`` together, so both the agility boundary and
        the momentum-storage boundary move with the sweep and the 7.5 s wheel time constant
        is preserved. See the module docstring, point 3.
    :param m_max: Per-axis dipole limit [A m^2] before ``authority_scale``.
    :param tau_w: Wheel torque limit [N m] before ``authority_scale``.
    :param h_max: Wheel momentum limit [N m s] before ``authority_scale``.
    :param m_res: Residual-dipole magnitude [A m^2]. Default 0.05 (reference bus); pass 0.1
        for the labelled sensitivity case.
    :param com_offset_m: cp-cg offset [m]. **Setting this to 0 zeroes the drag and SRP
        torques identically** -- see the module docstring, point 1.
    :param disturbances: Any of ``"gg"``, ``"drag"``, ``"srp"``, ``"dipole"``. Campaign B
        runs with all of them off.
    :param estimate_dipole: Carry the residual dipole as an augmented estimator state.
        Requires ``"dipole"`` in ``disturbances``.
    :param sensors: Any of ``"mtm"``, ``"gyro"``, ``"star_tracker"``, ``"sun"``.
    :param estimated: Return an :class:`EstimatedSatellite` (the filter's model) rather than
        the truth plant.
    :param J_com: Override the inertia **about the COM** (Campaign R matches genACS).
    :param mass: Override the mass [kg].
    :returns: :class:`Satellite` or :class:`EstimatedSatellite`.
    """
    if authority_scale <= 0.0:
        raise ValueError(f"authority_scale must be positive, got {authority_scale}")

    mass = IAC_6U.mass if mass is None else float(mass)
    m_max = (IAC_6U.m_max if m_max is None else float(m_max)) * authority_scale
    tau_w = (IAC_6U.tau_w if tau_w is None else float(tau_w)) * authority_scale
    h_max = (IAC_6U.h_max if h_max is None else float(h_max)) * authority_scale
    m_res = IAC_6U.m_res if m_res is None else float(m_res)
    com_offset_m = (IAC_6U.com_offset_m if com_offset_m is None
                    else float(com_offset_m))

    # --- Mass properties -----------------------------------------------------------
    com_dir = np.asarray(IAC_6U.com_offset_dir, dtype=float)
    com_dir = com_dir / np.linalg.norm(com_dir)
    COM = com_offset_m * com_dir
    J_com_mat = (np.diagflat(IAC_6U.J_com) if J_com is None
                 else np.asarray(J_com, dtype=float))
    if J_com_mat.ndim == 1:
        J_com_mat = np.diagflat(J_com_mat)
    J_0 = _J0_from_Jcom(J_com_mat, mass, COM)

    # --- Actuators -----------------------------------------------------------------
    acts: List[MTQ] = [MTQ(axis=_UV[j], max_torque=m_max) for j in range(3)]

    if wheel_axes is None:
        if n_rw == 0:
            axes: List[np.ndarray] = []
        elif n_rw == 1:
            axes = [np.asarray(IAC_6U.boresight, dtype=float)]  # boresight-aligned
        elif n_rw == 3:
            axes = [_UV[0], _UV[1], _UV[2]]
        else:
            raise ValueError(
                f"n_rw must be 0, 1 or 3 without explicit wheel_axes, got {n_rw}"
            )
    else:
        axes = [np.asarray(a, dtype=float) for a in wheel_axes]
    axes = [a / np.linalg.norm(a) for a in axes]

    # Tachometer noise: 1% of h_max, per the campaign spec's sensing table.
    rws: List[RW] = [
        RW(
            axis=a,
            max_torque=tau_w,
            J=IAC_6U.J_rw,
            h=0.0,
            h_max=h_max,
            h_meas_noise=Noise(noise=0.0, std_noise=0.01 * h_max),
            estimate_bias=False,
        )
        for a in axes
    ]
    acts.extend(rws)

    # --- Disturbances ---------------------------------------------------------------
    faces = iac_6u_geometry_faces()
    geom = GeometryConfig(faces)
    dist_list = []
    wanted = {d.lower() for d in disturbances}
    unknown = wanted - {"gg", "drag", "srp", "dipole"}
    if unknown:
        raise ValueError(f"unknown disturbance(s): {sorted(unknown)}")
    if "gg" in wanted:
        dist_list.append(GG_Disturbance())
    if "drag" in wanted:
        dist_list.append(Drag_Disturbance(geom))
    if "srp" in wanted:
        dist_list.append(SRP_Disturbance(geom))
    if "dipole" in wanted:
        m_dir = np.asarray(IAC_6U.m_res_dir, dtype=float)
        m_dir = m_dir / np.linalg.norm(m_dir)
        dist_list.append(
            Dipole_Disturbance(
                dipole_torque=m_res * m_dir,
                estimate_dist=bool(estimate_dipole),
            )
        )
    elif estimate_dipole:
        raise ValueError(
            "estimate_dipole=True requires 'dipole' in disturbances"
        )

    # --- Sensors ---------------------------------------------------------------------
    sens_list = []
    wanted_s = {s.lower() for s in sensors}
    unknown_s = wanted_s - {"mtm", "gyro", "star_tracker", "sun"}
    if unknown_s:
        raise ValueError(f"unknown sensor(s): {sorted(unknown_s)}")
    if "mtm" in wanted_s:
        sens_list += create_iac_magnetometer(estimate_bias=estimated)
    if "gyro" in wanted_s:
        sens_list += create_iac_gyro(estimate_bias=estimated)
    if "star_tracker" in wanted_s:
        sens_list.append(
            create_iac_star_tracker(boresight=np.asarray(IAC_6U.boresight, float))
        )
    if "sun" in wanted_s:
        sens_list += create_Clydespace_3U_array(axis=_UV[0], estimate_bias=estimated)
        sens_list += create_Clydespace_3U_array(axis=_UV[1], estimate_bias=estimated)

    cls = EstimatedSatellite if estimated else Satellite
    return cls(
        mass=mass,
        COM=COM,
        J_0=J_0,
        disturbances=dist_list,
        sensors=sens_list,
        actuators=acts,
        boresight=np.asarray(IAC_6U.boresight, dtype=float),
    )

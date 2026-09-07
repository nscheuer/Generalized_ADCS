"""Field-model error between the estimator's IGRF and the plant's (campaign §3).

The campaign retains ~4 degrees of direction error and ~4% of magnitude error between the
geomagnetic field the *onboard* software believes and the one the *plant* integrates, per
SSC26-FT-34 §IV-G. Without it, estimator and plant share one field and the field-model error
contributes **exactly zero** to the residual-dipole estimate -- which would make the Section IV
cancellation result optimistic in precisely the way the campaign spec warns against ("do not let
a good attitude solution disguise a poor dipole estimate").

The sim loop already provides the seam. In ``ADCS/simulate.py``::

    os_for_gnc = os_hat if os_hat is not None else os_k        # GNC path
    x_hat = estimator.update(u=u, sensors=y, os=os_for_gnc)    # plant integrates os_k

so perturbing the state handed to the GNC path is all that is needed; nothing in the framework
changes. ``papers/Planner/generate_p2.8_mismatch.py`` perturbs the plant *and* onboard sensing
together, with the planner on nominal -- the opposite split, and not what this campaign wants.

Error model
-----------
The perturbation is **fixed per trial**, not resampled per step: a field-model error is a
systematic error in IGRF coefficients and in the epoch the model is evaluated at, so it is
strongly correlated over an orbit. Resampling per step would average it away and understate its
effect on the dipole estimate -- the exact failure this module exists to prevent.

The rotation is taken about an axis **perpendicular to B**, so the realized angle between the
true and believed field is exactly :math:`\\theta` wherever the spacecraft is:

.. math::

    \\hat{n} = \\frac{\\mathbf{B} \\times \\hat{u}}{\\|\\mathbf{B} \\times \\hat{u}\\|},
    \\qquad
    \\mathbf{B}_{\\text{est}} = (1 + s)\\, R(\\hat{n}, \\theta)\\, \\mathbf{B}_{\\text{true}}

with :math:`\\hat{u}` a per-trial frozen reference vector that fixes which way the error leans
around **B**, :math:`\\theta` the direction error and :math:`s` the fractional magnitude error.

Rotating about a *fixed* axis instead would be defensible as physics -- a fixed coefficient error
does produce a position-dependent direction error -- but it delivers an **uncontrolled** error:
a rotation of :math:`\\theta` about a fixed axis moves a vector parallel to that axis not at all,
and one perpendicular to it by the full :math:`\\theta`. The realized error would then vary over
the orbit and average below the quoted figure, which is not what "retain ~4 degrees" asks for and
would be awkward to hold fixed across the Campaign E sweep.
"""

from __future__ import annotations

__all__ = ["FieldErrorModel", "wrap_os_for_gnc"]

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ADCS.helpers.math_helpers import rot_mat

_DEG = np.pi / 180.0


@dataclass
class FieldErrorModel:
    """A per-trial frozen rotation + scale applied to the GNC-side magnetic field.

    :param direction_deg: 1-sigma direction error [deg]. Campaign default 4.
    :param magnitude_frac: 1-sigma fractional magnitude error. Campaign default 0.04.
    :param rng: Draw source. Pass the trial's generator so the error is paired across
        conditions that share a seed.
    :param deterministic: If ``True``, use exactly ``direction_deg`` and ``magnitude_frac``
        rather than drawing from a normal with those sigmas. Useful for a sensitivity sweep
        over field error, where a known level is wanted rather than a random one.
    """

    direction_deg: float = 4.0
    magnitude_frac: float = 0.04
    rng: Optional[np.random.Generator] = None
    deterministic: bool = False

    def __post_init__(self) -> None:
        rng = np.random.default_rng() if self.rng is None else self.rng

        # Frozen reference that fixes which way the error leans around B. The rotation
        # axis itself is built per call as B x u, so it is always perpendicular to B.
        u = rng.standard_normal(3)
        self.reference = u / np.linalg.norm(u)

        if self.deterministic:
            self.theta = self.direction_deg * _DEG
            self.scale = 1.0 + self.magnitude_frac
        else:
            self.theta = float(rng.normal(0.0, self.direction_deg * _DEG))
            self.scale = 1.0 + float(rng.normal(0.0, self.magnitude_frac))

    def _axis_for(self, B_hat: np.ndarray) -> np.ndarray:
        """Unit rotation axis perpendicular to ``B_hat``, frozen in azimuth by ``reference``."""
        axis = np.cross(B_hat, self.reference)
        n = np.linalg.norm(axis)
        if n < 1e-8:
            # B is (nearly) parallel to the frozen reference; any perpendicular will do,
            # chosen deterministically so the model stays reproducible.
            alt = np.array([1.0, 0.0, 0.0])
            if abs(B_hat[0]) > 0.9:
                alt = np.array([0.0, 1.0, 0.0])
            axis = np.cross(B_hat, alt)
            n = np.linalg.norm(axis)
        return axis / n

    @property
    def realized_direction_deg(self) -> float:
        """Direction error actually drawn for this trial [deg] (report this, not the sigma)."""
        return abs(self.theta) / _DEG

    @property
    def realized_magnitude_frac(self) -> float:
        """Magnitude error actually drawn for this trial (report this, not the sigma)."""
        return self.scale - 1.0

    def apply(self, B: np.ndarray) -> np.ndarray:
        """Map a true field vector to what the onboard model believes.

        The realized angle between input and output is exactly ``|theta|``, independent of
        where in the orbit ``B`` points.
        """
        B = np.asarray(B, dtype=float).reshape(3)
        n = np.linalg.norm(B)
        if n < 1e-30:
            return self.scale * B

        axis = self._axis_for(B / n)
        q = np.concatenate(([np.cos(self.theta / 2.0)], axis * np.sin(self.theta / 2.0)))
        return self.scale * (rot_mat(q) @ B)


def wrap_os_for_gnc(os, model: FieldErrorModel):
    """Return a copy of ``os`` whose magnetic field carries the model error.

    Only ``B`` is replaced. Position, velocity, sun vector and density stay exactly as the
    plant sees them, because this models a *field-model* error, not a navigation error;
    orbit-determination error, if wanted, is a separate mechanism (``orbit_estimator``).

    ``B`` is the single source of truth for the field: ``Orbital_State.get_state_vector``
    derives the body-frame field from it on demand rather than caching one, so perturbing
    ``B`` is complete -- every downstream reader (sensor models, controllers, the estimator's
    predicted magnetometer measurement) sees a consistent perturbation.
    """
    out = os.copy()
    out.B = model.apply(out.B)
    return out

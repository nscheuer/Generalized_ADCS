__all__ = ["Actuator"]

import numpy as np
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.helpers.math_helpers import normalize

class Actuator:
    r"""
    Abstract actuator base class.

    **State & notation**

    - Body torque: :math:`\boldsymbol{\tau}\in\mathbb{R}^3`.
    - Command (input): :math:`u\in\mathbb{R}` with :math:`|u|\le u_\max`.
    - Body axis: :math:`\mathbf{a}=\mathrm{normalize}(\texttt{axis})\in\mathbb{S}^2`.
    - Base state: :math:`\mathbf{x}=[\,\boldsymbol{\omega}^\top,\ \mathbf{q}^\top\,]^\top\in\mathbb{R}^7`,
      where :math:`\boldsymbol{\omega}\in\mathbb{R}^3` and quaternion :math:`\mathbf{q}\in\mathbb{R}^4`.
    - Bias state and momentum-storage state are not modeled here (zero-size shapes).

    Subclasses should override the torque and derivative methods. The base
    implementation returns zeros with the documented shapes.
    """
    def __init__(self, axis: np.ndarray, u_max: float, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False) -> None:
        r"""
        Initialize the actuator with a normalized axis and input limit.

        Parameters
        ----------
        axis : (3,) array_like
            Actuation direction; normalized internally to :math:`\mathbf{a}\in\mathbb{S}^2`.
        u_max : float
            Maximum admissible command magnitude :math:`u_\max`.
        bias : Bias, optional
            Bias model; if provided, bias-related Jacobians mirror input Jacobians.
        noise : Noise, optional
            Noise model applied to outputs when requested.
        estimate_bias : bool, optional
            If ``True``, enables bias estimation logic (bookkeeping only here).
        """
        self.axis = normalize(axis)
        self.u_max = u_max
        if bias:
            self.bias: Bias = bias
        else:
            self.bias = Bias()
        if noise:
            self.noise: Noise = noise
        else:
            self.noise = Noise()
        self.estimate_bias: bool = estimate_bias
        self.last_bias_time: float = float('nan')

    def torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: DisturbanceMode = None) -> float:
        r"""
        Body-frame torque :math:`\boldsymbol{\tau}(u,\mathbf{x},\mathrm{os})`.

        Returns
        -------
        (3,) ndarray
            :math:`\boldsymbol{\tau}\in\mathbb{R}^3`. Base class returns :math:`\mathbf{0}`.

        Notes
        -----
        Subclasses typically implement
        :math:`\boldsymbol{\tau} = \boldsymbol{\tau}(u,\boldsymbol{\omega},\mathbf{q};\ \text{env}(\mathrm{os}))`.
        """
        return np.ndarray([0, 0, 0])
    
    def storage_torque(self, u: float, j2000: float, dmode: DisturbanceMode = None)-> float:
        r"""
        Momentum-storage torque contribution (e.g., reaction wheels).

        Returns
        -------
        (0,) ndarray
            Empty vector (no storage DOFs in this base class).
        """
        return np.zeros((0,))
    
    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative :math:`\partial\boldsymbol{\tau}/\partial u`.

        Returns
        -------
        (1, 3) ndarray
            Row-Jacobian w.r.t. scalar input :math:`u`.
        """
        return np.zeros((1, 3))
    
    def dtorq__dbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative :math:`\partial\boldsymbol{\tau}/\partial \text{bias}`.

        Returns
        -------
        (0, 3) or (1, 3) ndarray
            If a bias model exists, mirrors :meth:`dtorq__du`; otherwise empty with shape :math:`0\times 3`.
        """
        if self.bias:
            return self.dtorq__du(u=u, x=x, os=os)
        else:
            return np.zeros((0, 3))
        
    def dtorq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative :math:`\partial\boldsymbol{\tau}/\partial \mathbf{x}` with
        :math:`\mathbf{x}=[\boldsymbol{\omega};\mathbf{q}]`.

        Returns
        -------
        (7, 3) ndarray
            Row-Jacobian stacked as :math:`[\partial/\partial\boldsymbol{\omega};\ \partial/\partial\mathbf{q}]`.
        """
        return np.zeros((7, 3))
    
    def dtorq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative :math:`\partial\boldsymbol{\tau}/\partial \mathbf{h}` (storage state).

        Returns
        -------
        (0, 3) ndarray
            Empty in this base class (no storage state).
        """
        return np.zeros((0,3))
    
    def ddtorq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u^2`.

        Returns
        -------
        (1, 1, 3) ndarray
            Componentwise Hessian slices for each torque component.
        """
        return np.zeros((1, 1, 3))
    
    def ddtorq__dudbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u\,\partial\text{bias}`.

        Returns
        -------
        (1, 1, 3) or (1, 0, 3) ndarray
            If bias exists, mirrors :meth:`ddtorq__dudu`; else empty along bias dim.
        """
        if self.bias:
            return self.ddtorq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((1, 0, 3))
        
    def ddtorq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u\,\partial \mathbf{x}`.

        Returns
        -------
        (1, 7, 3) ndarray
            Stacked by :math:`(\boldsymbol{\omega},\mathbf{q})`.
        """
        return np.zeros((1, 7, 3))
    
    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u\,\partial \mathbf{h}`.

        Returns
        -------
        (1, 0, 3) ndarray
            Empty in this base class.
        """
        return np.zeros((1, 0, 3))
    
    def ddtorq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial\text{bias}^2`.

        Returns
        -------
        (1, 1, 3) or (0, 0, 3) ndarray
            If bias exists, mirrors :meth:`ddtorq__dudu`; else empty.
        """
        if self.bias:
            return self.ddtorq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial\text{bias}\,\partial \mathbf{x}`.

        Returns
        -------
        (1, 7, 3) or (0, 7, 3) ndarray
            If bias exists, mirrors :meth:`ddtorq__dudbasestate`; else empty along bias dim.
        """
        if self.bias:
            return self.ddtorq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 3))
        
    def ddtorq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial\text{bias}\,\partial \mathbf{h}`.

        Returns
        -------
        (1, 0, 3) or (0, 0, 3) ndarray
            If bias exists, mirrors :meth:`ddtorq__dudh`; else empty.
        """
        if self.bias:
            return self.ddtorq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State):
        return np.zeros((7,7,3))
        
    def ddtorq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{x}\,\partial \mathbf{h}`.

        Returns
        -------
        (7, 0, 3) ndarray
            Empty in this base class.
        """
        return np.zeros((7, 0, 3))
    
    def ddtorq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{h}^2`.

        Returns
        -------
        (0, 0, 3) ndarray
            Empty in this base class.
        """
        return np.zeros((0, 0, 3))
    
    def dstor_torq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Derivative of storage torque w.r.t. input: :math:`\partial \mathbf{t}_s/\partial u`.

        Returns
        -------
        (1, 0) ndarray
            Empty because :math:`\mathbf{t}_s\in\mathbb{R}^0` here.
        """
        return np.zeros((1, 0))
    
    def dstor_torq__dbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Derivative :math:`\partial \mathbf{t}_s/\partial \text{bias}`.

        Returns
        -------
        (1, 0) or (0, 0) ndarray
            If bias exists, mirrors :meth:`dstor_torq__du`; else empty.
        """
        if self.bias:
            return self.dstor_torq__du(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0))
        
    def dstor_torq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Derivative :math:`\partial \mathbf{t}_s/\partial \mathbf{x}`.

        Returns
        -------
        (7, 0) ndarray
            Empty in this base class.
        """
        return np.zeros((7, 0))
        
    def dstor_torq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Derivative :math:`\partial \mathbf{t}_s/\partial \mathbf{h}`.

        Returns
        -------
        (0, 0) ndarray
            Empty in this base class.
        """
        return np.zeros((0, 0))
    
    def ddstor_torq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2 \mathbf{t}_s/\partial u^2`.

        Returns
        -------
        (1, 1, 0) ndarray
            Empty last dimension (no storage DOFs).
        """
        return np.zeros((1, 1, 0))
    
    def ddstor_torq__dudbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial u\,\partial\text{bias}`.

        Returns
        -------
        (1, 1, 0) or (1, 0, 0) ndarray
            If bias exists, mirrors :meth:`ddstor_torq__dudu`; else empty along bias dim.
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((1, 0, 0))

    def ddstor_torq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial u\,\partial \mathbf{x}`.

        Returns
        -------
        (1, 7, 0) ndarray
            Empty last dimension.
        """
        return np.zeros((1, 7, 0))
    
    def ddstor_torq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial u\,\partial \mathbf{h}`.

        Returns
        -------
        (1, 0, 0) ndarray
            Empty in this base class.
        """
        return np.zeros((1, 0, 0))
    
    def ddstor_torq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2 \mathbf{t}_s/\partial\text{bias}^2`.

        Returns
        -------
        (1, 1, 0) or (0, 0, 0) ndarray
            If bias exists, mirrors :meth:`ddstor_torq__dudu`; else empty.
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial\text{bias}\,\partial \mathbf{x}`.

        Returns
        -------
        (1, 7, 0) or (0, 7, 0) ndarray
            If bias exists, mirrors :meth:`ddstor_torq__dudbasestate`; else empty along bias dim.
        """
        if self.bias:
            return self.ddstor_torq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 0))
        
    def ddstor_torq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial\text{bias}\,\partial \mathbf{h}`.

        Returns
        -------
        (1, 0, 0) or (0, 0, 0) ndarray
            If bias exists, mirrors :meth:`ddstor_torq__dudh`; else empty.
        """
        if self.bias:
            return self.ddstor_torq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2 \mathbf{t}_s/\partial \mathbf{x}\,\partial \mathbf{x}`.

        Returns
        -------
        (7, 7, 0) ndarray
            Empty last dimension.
        """
        return np.zeros((7, 7, 0))
    
    def ddstor_torq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed :math:`\partial^2 \mathbf{t}_s/\partial \mathbf{x}\,\partial \mathbf{h}`.

        Returns
        -------
        (7, 0, 0) ndarray
            Empty in this base class.
        """
        return np.zeros((7, 0, 0))
    
    def ddstor_torq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2 \mathbf{t}_s/\partial \mathbf{h}^2`.

        Returns
        -------
        (0, 0, 0) ndarray
            Empty in this base class.
        """
        return np.zeros((0, 0, 0))
    


    

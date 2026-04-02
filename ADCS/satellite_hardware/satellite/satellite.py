__all__ = ["Satellite"]

import numpy as np
from numba import njit

from typing import List, Dict, Union, Tuple, Any
from scipy.linalg import block_diag
import time

import ADCS.orbits.universal_constants as uc
from ADCS.helpers.math_helpers import *
from ADCS.satellite_hardware.disturbances import Disturbance, SRP_Disturbance, General_Disturbance, Prop_Disturbance
from ADCS.satellite_hardware.sensors import Sensor, GPS, Gyro, MTM, SunPair, SunSensor
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


@njit(cache=True)
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _quat_qdot(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]

    return 0.5 * np.array(
        [
            -(w0 * q1 + w1 * q2 + w2 * q3),
            q0 * w0 + q2 * w2 - q3 * w1,
            q0 * w1 + q3 * w0 - q1 * w2,
            q0 * w2 + q1 * w1 - q2 * w0,
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _wdot_no_rw_kernel(
    w: np.ndarray, total_torque: np.ndarray, J: np.ndarray, invJ_noRW: np.ndarray
) -> np.ndarray:
    Jw = w @ J
    return (-_cross3(w, Jw) + total_torque) @ invJ_noRW


@njit(cache=True)
def _wdot_rw_kernel(
    w: np.ndarray,
    h: np.ndarray,
    total_torque: np.ndarray,
    J: np.ndarray,
    invJ_noRW: np.ndarray,
    rw_axes: np.ndarray,
) -> np.ndarray:
    coupled_momentum = w @ J + h @ rw_axes
    return (-_cross3(w, coupled_momentum) + total_torque) @ invJ_noRW


@njit(cache=True)
def _rw_hdot_kernel(u_rw: np.ndarray, wdot: np.ndarray, rw_axes: np.ndarray, rw_js: np.ndarray) -> np.ndarray:
    return u_rw - (wdot @ rw_axes.T) * rw_js


@njit(cache=True)
def _sum_torques(torques: np.ndarray) -> np.ndarray:
    """Aggregate torque vectors (n, 3) into a single (3,) vector."""
    total = np.zeros(3, dtype=np.float64)
    for i in range(torques.shape[0]):
        total = total + torques[i, :]
    return total


_DEFAULT_DMODE = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
_SOLVER_DMODE = ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False)
_NOISELESS_DMODE = ErrorMode(add_bias=False, add_noise=False, update_bias=False, update_noise=False)

class Satellite:
    r"""
    Rigid-body spacecraft model with sensors, actuators, and environmental disturbances.

    This class bundles the spacecraft **physical properties** (mass, center of mass, inertia),
    a set of **actuators** (e.g., :class:`~ADCS.satellite_hardware.actuators.RW`,
    :class:`~ADCS.satellite_hardware.actuators.MTQ`), **sensors**
    (e.g., :class:`~ADCS.satellite_hardware.sensors.Gyro`,
    :class:`~ADCS.satellite_hardware.sensors.MTM`,
    :class:`~ADCS.satellite_hardware.sensors.SunSensor`,
    :class:`~ADCS.satellite_hardware.sensors.GPS`), and **disturbance models**
    (e.g., :class:`~ADCS.satellite_hardware.disturbances.SRP_Disturbance`).

    The rotational state is represented by body angular velocity and a unit quaternion.
    Reaction wheel angular momentum states may be appended.

    **State definition**

    Let the full state be

    .. math::

        \mathbf{x} \;=\;
        \begin{bmatrix}
        \boldsymbol{\omega} \\
        \mathbf{q} \\
        \mathbf{h}_{\mathrm{RW}}
        \end{bmatrix}
        \in \mathbb{R}^{7+n_{\mathrm{RW}}},

    where

    * :math:`\boldsymbol{\omega}\in\mathbb{R}^3` is the body angular velocity (rad/s),
    * :math:`\mathbf{q}\in\mathbb{R}^4` is a unit quaternion (Hamilton convention, body→ECI),
    * :math:`\mathbf{h}_{\mathrm{RW}}\in\mathbb{R}^{n_{\mathrm{RW}}}` are wheel momenta.

    **Kinematics**

    Quaternion kinematics are

    .. math::

        \dot{\mathbf{q}} = \frac{1}{2}\,\Omega(\boldsymbol{\omega})\,\mathbf{q},

    with an equivalent matrix form implemented via :func:`~ADCS.helpers.math_helpers.Wmat`.

    **Dynamics (bus + wheels)**

    Define the total external torque

    .. math::

        \boldsymbol{\tau} \;=\; \boldsymbol{\tau}_{\mathrm{dist}}(\mathbf{x}, \mathrm{os}) \;+\;
        \boldsymbol{\tau}_{\mathrm{act}}(\mathbf{x}, \mathbf{u}, \mathrm{os}).

    The rigid-body rotational equation is

    .. math::

        \mathbf{J}\,\dot{\boldsymbol{\omega}}
        \;=\;
        -\boldsymbol{\omega}\times\Big(\mathbf{J}\boldsymbol{\omega} + \mathbf{H}_{\mathrm{RW}}\Big)
        \;+\; \boldsymbol{\tau},

    where :math:`\mathbf{H}_{\mathrm{RW}}` is the body-frame contribution of stored wheel momentum.
    When wheels exist, wheel momentum dynamics follow actuator storage torques and the coupling term
    induced by :math:`\dot{\boldsymbol{\omega}}`.

    **Frames and environment**

    Environmental vectors are provided by :class:`~ADCS.orbits.orbital_state.Orbital_State`
    and rotated into the body frame via :func:`~ADCS.helpers.math_helpers.rot_mat`.

    **Inertia shifting**

    The inertia tensor about the center of mass is computed using the parallel axis theorem:

    .. math::

        \mathbf{J}_{\mathrm{COM}}
        \;=\;
        \mathbf{J}_0 \;-\; m\left(\|\mathbf{r}_{\mathrm{COM}}\|^2\mathbf{I}
        \;-\;\mathbf{r}_{\mathrm{COM}}\mathbf{r}_{\mathrm{COM}}^\top\right),

    where :math:`\mathbf{J}_0` is the inertia about the reference origin and
    :math:`\mathbf{r}_{\mathrm{COM}}` is the COM offset in the same frame.

    .. note::
        Many methods accept an :class:`~ADCS.satellite_hardware.errors.ErrorMode` to control
        bias/noise injection for Monte-Carlo simulation vs. deterministic propagation.

    .. rubric:: Stored attributes (selected)

    +-------------------+--------------------------------+
    | Attribute         | Meaning                        |
    +===================+================================+
    | ``mass``          | Total mass (kg)                |
    +-------------------+--------------------------------+
    | ``COM``           | Center of mass vector (m)      |
    +-------------------+--------------------------------+
    | ``J_0``           | Inertia about reference (kg·m²)|
    +-------------------+--------------------------------+
    | ``J_COM``         | Inertia about COM (kg·m²)      |
    +-------------------+--------------------------------+
    | ``J_noRW``        | Bus inertia excluding wheels   |
    +-------------------+--------------------------------+
    | ``state_len``     | ``7 + number_RW``              |
    +-------------------+--------------------------------+
    | ``control_len``   | ``len(actuators)``             |
    +-------------------+--------------------------------+

    :raises ValueError:
        If ``COM`` is not shape ``(3,)`` or ``J_0`` is not shape ``(3,3)``.
    """
    def __init__(self, mass: float = 1.0, COM: np.ndarray = None, J_0: np.ndarray = None, disturbances: List[Disturbance] = [], sensors: List[Sensor] = [], actuators: List[Actuator] = [], boresight: dict[str, np.ndarray] | np.ndarray = None) -> None:
        r"""
        Construct a :class:`~ADCS.satellite.Satellite`.

        Initializes mass/inertia properties, categorizes sensors and actuators by type,
        computes derived inertia matrices via :meth:`~ADCS.satellite.Satellite.update_J`,
        and determines state/control vector dimensions.

        :param mass: Total satellite mass including angular momentum storage hardware (kg).
        :type mass: float

        :param COM: Center of mass vector in the body/reference frame, shape ``(3,)``.
            If ``None``, defaults to ``[0,0,0]``.
        :type COM: numpy.ndarray | None

        :param J_0: Inertia tensor about the reference origin, shape ``(3,3)`` (kg·m²).
            If ``None``, defaults to identity.
        :type J_0: numpy.ndarray | None

        :param disturbances: List of disturbance models (e.g., SRP, drag, gravity gradient).
        :type disturbances: list[:class:`~ADCS.satellite_hardware.disturbances.Disturbance`]

        :param sensors: List of sensor models.
        :type sensors: list[:class:`~ADCS.satellite_hardware.sensors.Sensor`]

        :param actuators: List of actuator models.
        :type actuators: list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]

        :param boresight: Nominal body-frame boresight direction, shape ``(3,)``.
        :type boresight: dict[str, numpy.ndarray]

        :return: ``None``.
        :rtype: None

        :raises ValueError:
            If ``COM`` is not shape ``(3,)`` or ``J_0`` is not shape ``(3,3)``.
        """
        self.mass = mass # Includes angular momentum storage
        self.COM = np.asarray(COM, dtype=float) # Includes angular momentum storage
        if COM is None:
            self.COM = np.zeros(3)
        else:
            self.COM = np.asarray(COM, dtype=float)
            if self.COM.shape != (3,):
                raise ValueError(f"COM must be a numpy array of shape (3,), got {self.COM.shape}")

        if J_0 is None:
            self.J_0 = np.eye(3)
        else:
            self.J_0 = np.asarray(J_0, dtype=float)
            if self.J_0.shape != (3, 3):
                raise ValueError(f"J must be a numpy array of shape (3, 3), got {self.J_0.shape}")
        self.disturbances = disturbances
        self.sensors = sensors
        self.attitude_sensors: List[Sensor] = [s for s in sensors if not isinstance(s, GPS)]
        self.GPS_sensors: List[GPS] = [s for s in sensors if isinstance(s, GPS)]
        
        self.actuators = actuators
        self.rw_actuators: List[RW] = [s for s in actuators if isinstance(s, RW)]
        self.mtq_actuators: List[MTQ] = [s for s in actuators if not isinstance(s, RW)]
        self.momentum_inds = np.array([j for j in range(len(self.actuators)) if isinstance(self.actuators[j], RW)])
        
        if boresight is None:
            self.boresight: dict[str, np.ndarray] = {
                "default": np.array([0, 0, 1], dtype=float)
            }
        elif isinstance(boresight, dict):
            if len(boresight) == 0:
                raise ValueError("Boresight dictionary cannot be empty")
            self.boresight = {}
            for name, vec in boresight.items():
                vec = np.asarray(vec, dtype=float)
                if vec.shape != (3,):
                    raise ValueError(f"Boresight vector '{name}' must be shape (3,), got {vec.shape}")
                self.boresight[str(name)] = normalize(vec)
        else:
            vec = np.asarray(boresight, dtype=float)
            if vec.shape != (3,):
                raise ValueError(f"Boresight vector must be shape (3,), got {vec.shape}")
            self.boresight = {"default": normalize(vec)}
            
        self.number_RW = sum([1 for j in self.actuators if isinstance(j,RW)])

        # Initialize state
        self.state_len = 7 + self.number_RW
        self.control_len = len(actuators)
        self._rw_axes_cached = np.zeros((0, 3), dtype=float)
        self._rw_js_cached = np.zeros(0, dtype=float)
        self.update_J(J_0=J_0, COM=COM)

    def get_boresight(self, name: str | None = None) -> np.ndarray:
        if name is None:
            name = list(self.boresight.keys())[0]

        if name not in self.boresight:
            raise KeyError(
                f"Unknown boresight '{name}'. Available: {list(self.boresight.keys())}"
            )

        return self.boresight[name]

    def update_J(self, J_0: np.ndarray = None, COM: np.ndarray = None) -> None:
        r"""
        Update inertia matrices and cached inverses.

        This method validates the inertia tensor, enforces symmetry, checks positive definiteness,
        shifts the inertia to the center of mass, and computes additional derived inertias.

        **Validation**

        The inertia is required to be real, symmetric, and positive definite:

        .. math::

            \mathbf{J} = \mathbf{J}^\top,\qquad \lambda_i(\mathbf{J})>0.

        **Parallel axis theorem**

        If :math:`\mathbf{J}_0` is about the reference origin and :math:`\mathbf{r}_{COM}` is the COM offset,
        then

        .. math::

            \mathbf{J}_{COM}
            \;=\;
            \mathbf{J}_0
            \;-\;
            m\left(\|\mathbf{r}_{COM}\|^2\mathbf{I} - \mathbf{r}_{COM}\mathbf{r}_{COM}^\top\right).

        **Excluding reaction wheel inertia**

        If wheels are present, the bus-only inertia is approximated by subtracting wheel spin-axis
        contributions:

        .. math::

            \mathbf{J}_{noRW} \;=\; \mathbf{J}_{COM} \;-\;
            \sum_{k=1}^{n_{RW}} J_{RW,k}\,\hat{\mathbf{a}}_k \hat{\mathbf{a}}_k^\top,

        where :math:`J_{RW,k}` and :math:`\hat{\mathbf{a}}_k` are the wheel rotor inertia and spin axis.

        Cached inverses are computed for efficient dynamics evaluation.

        :param J_0: Inertia tensor about the reference origin, shape ``(3,3)``. If ``None``, uses ``self.J_0``.
        :type J_0: numpy.ndarray | None

        :param COM: Center of mass offset, shape ``(3,)``. If ``None``, uses ``self.COM``.
        :type COM: numpy.ndarray | None

        :return: ``None``.
        :rtype: None

        :raises ValueError:
            If ``J_0`` cannot be reshaped to ``(3,3)``, contains non-real values,
            is not symmetric within tolerance, or is not positive definite.
        """
        if J_0 is None: J_0 = self.J_0
        if COM is None: COM = self.COM

        try:
            J_0 = np.array(J_0, dtype=float).reshape((3,3))
        except:
            raise ValueError("J must be convertible to a (3,3) array")
        
        # Physical validity checks
        if not np.all(np.isreal(J_0)):
            raise ValueError("J contains non-real values")
        if not np.allclose(J_0, J_0.T, rtol=1e-5, atol=1e-8):
            raise ValueError("J must be symmetric")
        J = 0.5 * (J_0 + J_0.T)  # enforce symmetry

        eigvals = np.linalg.eigvals(J)
        if not np.all(eigvals > 0):
            print("Inertia eigenvalues:", eigvals)
            raise ValueError("J must be positive definite")
        
        self.J_0 = J_0
        self.invJ_0 = np.linalg.inv(J)

        # Apply the parallel axis theorem to move to COM
        self.J_COM = J - self.mass * (np.eye(3) * np.dot(COM, COM) - np.outer(COM, COM))
        self.invJ_COM = np.linalg.inv(self.J_COM)

        # Subtract reaction wheel contributions (if applicable)
        self.J_noRW = self.J_COM - np.sum(np.array([rw.J*np.outer(rw.axis, rw.axis) for rw in self.rw_actuators]), axis = 0)
        # self.J_noRW = self.J_COM - np.sum(
        #     np.array([
        #         self.actuators[j].J * np.outer(self.actuators[j].axis, self.actuators[j].axis)
        #         for j in getattr(self, "momentum_inds", [])
        #     ]),
        #     axis=0
        # ) if getattr(self, "momentum_inds", None) else self.J_COM

        self.invJ_noRW = np.linalg.inv(self.J_noRW)

        if self.number_RW > 0:
            self._rw_axes_cached = np.vstack([self.actuators[j].axis for j in self.momentum_inds]).astype(float)
            self._rw_js_cached = np.array([self.actuators[j].J for j in self.momentum_inds], dtype=float)
        else:
            self._rw_axes_cached = np.zeros((0, 3), dtype=float)
            self._rw_js_cached = np.zeros(0, dtype=float)
        
        # Cache dispatch maps for faster torque aggregation
        self._mtq_inds = np.array([j for j in range(len(self.actuators)) if not isinstance(self.actuators[j], RW)], dtype=int)
        self._rw_inds = np.array([j for j in range(len(self.actuators)) if isinstance(self.actuators[j], RW)], dtype=int)
        
        # Pre-check which disturbances accept sat parameter for fast dispatch
        self._dist_needs_sat = np.array([
            'sat' in d.torque.__code__.co_varnames for d in self.disturbances
        ], dtype=bool)

    def _toggle_disturbance(self, dist_class: Disturbance, on: bool, ind: int | None = None) -> None:
        r"""
        Enable or disable disturbance model(s) by type and optional index.

        If ``ind`` is provided, only the disturbance at that index is toggled and must be an instance of
        ``dist_class``. Otherwise, all registered disturbances of that type are toggled.

        This method calls each model's ``turn_on()`` / ``turn_off()`` methods.

        :param dist_class: Disturbance class to match against.
        :type dist_class: type[:class:`~ADCS.satellite_hardware.disturbances.Disturbance`]

        :param on: If ``True``, turn on. If ``False``, turn off.
        :type on: bool

        :param ind: Optional index into ``self.disturbances`` to toggle a single disturbance.
        :type ind: int | None

        :return: ``None``.
        :rtype: None

        :raises ValueError:
            If ``ind`` is given and the disturbance at ``ind`` is not an instance of ``dist_class``.
        """
        if ind is not None:
            d = self.disturbances[ind]
            if not isinstance(d, dist_class):
                raise ValueError(
                    f"Disturbance at index {ind} is not of type {dist_class.__name__}"
                )
            getattr(d, "turn_on" if on else "turn_off")()
            return

        # Otherwise apply to all disturbances of that type
        for d in self.disturbances:
            if isinstance(d, dist_class):
                getattr(d, "turn_on" if on else "turn_off")()

    def srp_dist_on(self) -> None:
        r"""
        Turn on all Solar Radiation Pressure disturbances.

        Equivalent to calling:

        .. code-block:: python

            self._toggle_disturbance(SRP_Disturbance, on=True)

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(SRP_Disturbance, on=True)

    def srp_dist_off(self) -> None:
        r"""
        Turn off all Solar Radiation Pressure disturbances.

        Equivalent to calling:

        .. code-block:: python

            self._toggle_disturbance(SRP_Disturbance, on=False)

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(SRP_Disturbance, on=False)

    def gen_dist_on(self, ind: int | None = None) -> None:
        r"""
        Turn on general disturbances.

        General disturbances correspond to :class:`~ADCS.satellite_hardware.disturbances.General_Disturbance`.

        :param ind: Optional index to toggle only one disturbance instance.
        :type ind: int | None

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(General_Disturbance, on=True, ind=ind)

    def gen_dist_off(self, ind: int | None = None) -> None:
        r"""
        Turn off general disturbances.

        General disturbances correspond to :class:`~ADCS.satellite_hardware.disturbances.General_Disturbance`.

        :param ind: Optional index to toggle only one disturbance instance.
        :type ind: int | None

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(General_Disturbance, on=False, ind=ind)

    def prop_dist_on(self, ind: int | None = None) -> None:
        r"""
        Turn on propulsion disturbances.

        Propulsion disturbances correspond to :class:`~ADCS.satellite_hardware.disturbances.Prop_Disturbance`.

        :param ind: Optional index to toggle only one disturbance instance.
        :type ind: int | None

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(Prop_Disturbance, on=True, ind=ind)

    def prop_dist_off(self, ind: int | None = None) -> None:
        r"""
        Turn off propulsion disturbances.

        Propulsion disturbances correspond to :class:`~ADCS.satellite_hardware.disturbances.Prop_Disturbance`.

        :param ind: Optional index to toggle only one disturbance instance.
        :type ind: int | None

        :return: ``None``.
        :rtype: None
        """
        self._toggle_disturbance(Prop_Disturbance, on=False, ind=ind)

    def specific_dist_on(self, ind: int) -> None:
        r"""
        Turn on a specific disturbance model by index.

        :param ind: Index into ``self.disturbances``.
        :type ind: int

        :return: ``None``.
        :rtype: None

        :raises IndexError:
            If ``ind`` is out of range.
        """
        self.disturbances[ind].turn_on()

    def specific_dist_off(self, ind: int) -> None:
        r"""
        Turn off a specific disturbance model by index.

        :param ind: Index into ``self.disturbances``.
        :type ind: int

        :return: ``None``.
        :rtype: None

        :raises IndexError:
            If ``ind`` is out of range.
        """
        self.disturbances[ind].turn_off()

    def RWhs(self) -> np.ndarray:
        r"""
        Return the current reaction wheel momenta as a vector.

        The returned vector is ordered according to ``self.momentum_inds`` (indices of
        :class:`~ADCS.satellite_hardware.actuators.RW` within ``self.actuators``).

        :return: Reaction wheel momentum vector, shape ``(number_RW,)``.
        :rtype: numpy.ndarray
        """
        return np.array([self.actuators[j].h for j in self.momentum_inds])

    def update_RWhs(self,state_or_RWhs) -> None:
        r"""
        Update stored wheel momentum values in each :class:`~ADCS.satellite_hardware.actuators.RW`.

        If ``state_or_RWhs`` has length ``self.state_len``, it is interpreted as the full state
        :math:`\mathbf{x}` and wheel momenta are extracted via
        :meth:`~ADCS.satellite.Satellite.RWhs_from_state`. Otherwise, it is interpreted as the
        wheel momentum vector directly.

        :param state_or_RWhs: Either full state vector (shape ``(state_len,)``) or wheel momenta
            (shape ``(number_RW,)``).
        :type state_or_RWhs: numpy.ndarray

        :return: ``None``.
        :rtype: None

        :raises ValueError:
            If the inferred wheel momentum vector length does not equal ``self.number_RW``.
        """
        if np.size(state_or_RWhs) == self.state_len:
            RWhs = self.RWhs_from_state(state_or_RWhs)
        else:
            RWhs = state_or_RWhs
        if np.size(RWhs) != self.number_RW:
            raise ValueError("wrong number of RWhs to update")
        [self.actuators[self.momentum_inds[i]].update_momentum(RWhs[i]) for i in range(len(self.momentum_inds))]

    def RWhs_from_state(self,state) -> np.ndarray:
        r"""
        Extract the reaction wheel momenta subvector from a full state vector.

        Using the state layout

        .. math::

            \mathbf{x} =
            \begin{bmatrix}
            \boldsymbol{\omega} & \mathbf{q} & \mathbf{h}_{RW}
            \end{bmatrix}^\top,

        this returns :math:`\mathbf{h}_{RW} = \mathbf{x}[7:]`.

        :param state: Full state vector, shape ``(state_len,)``.
        :type state: numpy.ndarray

        :return: Reaction wheel momentum vector, shape ``(number_RW,)``.
        :rtype: numpy.ndarray
        """
        return state[7:]
    
    def dynamics_core(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State, dmode: ErrorMode = None, verbose: bool = False) -> np.ndarray:
        r"""
        Continuous-time attitude dynamics :math:`\dot{\mathbf{x}} = f(\mathbf{x},\mathbf{u},\mathrm{os})`.

        This is the primary nonlinear dynamics model used by propagation, simulation,
        and linearization routines.

        **Kinematics**

        Quaternion kinematics are computed as

        .. math::

            \dot{\mathbf{q}}
            \;=\;\frac{1}{2}\,W(\mathbf{q})^\top\,\boldsymbol{\omega},

        where :math:`W(\mathbf{q})` is produced by :func:`~ADCS.helpers.math_helpers.Wmat`.

        **Torques**

        Disturbance and actuator torques are summed:

        .. math::

            \boldsymbol{\tau}_{dist} = \sum_i \boldsymbol{\tau}_{dist,i}(\mathbf{x},\mathrm{os}),
            \qquad
            \boldsymbol{\tau}_{act} = \sum_j \boldsymbol{\tau}_{act,j}(\mathbf{x},\mathbf{u},\mathrm{os}),

        using :meth:`~ADCS.satellite.Satellite.dist_torques` and
        :meth:`~ADCS.satellite.Satellite.act_torque`.

        **Rigid-body dynamics**

        If no reaction wheels are present:

        .. math::

            \dot{\boldsymbol{\omega}}
            \;=\;\mathbf{J}_{noRW}^{-1}\Big(
            -\boldsymbol{\omega}\times(\mathbf{J}\boldsymbol{\omega}) + \boldsymbol{\tau}
            \Big).

        If reaction wheels are present, let :math:`\mathbf{A}\in\mathbb{R}^{n_{RW}\times 3}`
        stack wheel spin axes as rows, and let :math:`\mathbf{h}\in\mathbb{R}^{n_{RW}}`
        be wheel momenta. Then the coupled term uses the body-frame stored momentum
        :math:`\mathbf{H}_{RW} = \mathbf{h}^\top \mathbf{A}` and

        .. math::

            \dot{\boldsymbol{\omega}}
            \;=\;\mathbf{J}_{noRW}^{-1}\Big(
            -\boldsymbol{\omega}\times(\mathbf{J}\boldsymbol{\omega} + \mathbf{H}_{RW})
            + \boldsymbol{\tau}
            \Big).

        Wheel momentum dynamics use the actuator storage torque commands
        (from each :class:`~ADCS.satellite_hardware.actuators.RW`) and subtract the
        coupling term from body acceleration:

        .. math::

            \dot{\mathbf{h}}
            \;=\;\mathbf{u}_{RW} \;-\; \mathbf{A}\,\dot{\boldsymbol{\omega}}\,\mathrm{diag}(J_{RW}),

        matching the implementation structure.

        :param x: State vector ``[w(3), q(4), h(number_RW)]``, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Control vector, one element per actuator, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param orbital_state: Orbital/environmental state providing vectors (e.g. magnetic field),
            used by torques.
        :type orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Error/noise/bias mode used by sensors/actuators/disturbances. If ``None``,
            a default with bias+noise enabled and updated is used.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` | None

        :param verbose: If ``True``, prints a detailed torque and derivative breakdown.
        :type verbose: bool

        :return: Time derivative vector :math:`\dot{\mathbf{x}}`, shape ``(state_len,)``.
        :rtype: numpy.ndarray
        """
        w = x[0:3]
        q = x[3:7]
        h = x[7:]
        J = self.J_0
        invJ_noRW = self.invJ_noRW

        if dmode is None:
            dmode = _DEFAULT_DMODE

        disturbance_torque: np.ndarray = self.dist_torques(x=x, os=orbital_state, dmode=dmode)
        actuator_torque: np.ndarray = self.act_torque(x=x, u=u, os=orbital_state, dmode=dmode)

        # Dynamics
        qdot = _quat_qdot(w, q)
        total_torque = disturbance_torque + actuator_torque

        # Reaction wheels
        if self.number_RW==0:
            wdot = _wdot_no_rw_kernel(w, total_torque, J, invJ_noRW)
            result = np.concatenate([wdot,qdot])
        else:
            RWjs = self._rw_js_cached
            RWaxes = self._rw_axes_cached
            storage_torques = [self.actuators[j].storage_torque(u=u[j], x=x, os=orbital_state, dmode=dmode) for j in self.momentum_inds]
            u_RW = np.array(storage_torques)
            wdot = _wdot_rw_kernel(w, h, total_torque, J, invJ_noRW, RWaxes)
            RW_hdot = _rw_hdot_kernel(u_RW, wdot, RWaxes, RWjs)

            result = np.concatenate([wdot,qdot,RW_hdot])

        if verbose:
            num_mtq = len(self.mtq_actuators)
            num_rw = len(self.rw_actuators)
            num_magic = 0
            print(f"\nInfo - Num MTQ: {num_mtq}, Num RW: {num_rw}, Num Magic: {num_magic}")#
            print(f"Input u: {u}")
            print(f"State w: {w}")
            print(f"State q: {q}")
            if num_rw > 0:
                print(f"State h: {h}")
            B_body = orbital_state.get_state_vector(x=x)["b"]
            print(f"Mag Field (Body): {B_body}")
            print("--- Torques ---")
            print(f"Calculated Mag Dipole (magvec): {u[:num_mtq]}")
            mtq_torque = sum([self.mtq_actuators[i].torque(u=u[i], x=x, os=orbital_state, dmode=dmode) for i in range(num_mtq)], np.zeros(3))
            print(f"Torque from Mag (m x B): {mtq_torque}")
            rw_torque = sum([self.rw_actuators[i].torque(u=u[num_mtq + i], x=x, os=orbital_state, dmode=dmode) for i in range(num_rw)], np.zeros(3))
            print(f"Torque from RW: {rw_torque}")
            print(f"Torque from Gyro (w x Jw): {np.cross(w,w@J+ h@RWaxes)}")
            print(f"Torque from Disturabnce: {disturbance_torque}")
            print(f"Total Net Torque: {total_torque-np.cross(w,w@J+ h@RWaxes)}\n")
            
            print("--- Derivatives ---")
            print(f"wdot: {wdot}")
            print(f"xdot (res): {result}")

        return result
        
    def dynamics_for_solver(self, t: float, x: np.ndarray, u: np.ndarray, os0: Orbital_State, os1: Orbital_State) -> np.ndarray:
        r"""
        Wrapper for ODE solvers expecting signature ``f(t, x)``.

        This method interpolates the orbital/environmental state between ``os0`` and ``os1`` using
        the fraction of elapsed time

        .. math::

            \alpha = \frac{t}{\Delta t},\qquad
            \Delta t = (J2000_1 - J2000_0)\cdot C_{\mathrm{cent}\to \mathrm{sec}},

        and calls :meth:`~ADCS.satellite.Satellite.dynamics_core` with an
        :class:`~ADCS.satellite_hardware.errors.ErrorMode` that **does not update** bias/noise
        so the solver uses a consistent noise sample.

        :param t: Time since start of the interval (s).
        :type t: float

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Constant (or externally provided) control input for the interval, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param os0: Orbital state at the interval start.
        :type os0: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param os1: Orbital state at the interval end.
        :type os1: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: State derivative :math:`\dot{\mathbf{x}}`, shape ``(state_len,)``.
        :rtype: numpy.ndarray
        """
        dmode = _SOLVER_DMODE
        delta_t_between_os = (os1.J2000 - os0.J2000)*TimeConstants.cent2sec
        time_frac = t/delta_t_between_os

        os = os0.average(os1, time_frac, True)

        x_dot = self.dynamics_core(x=x, u=u, orbital_state=os, dmode=dmode, verbose=False)
        
        return x_dot
    

    def dist_torques(self, x: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> np.ndarray:
        r"""
        Compute the total disturbance torque.

        The total disturbance torque is the sum over all registered disturbances:

        .. math::

            \boldsymbol{\tau}_{dist}(\mathbf{x},\mathrm{os})
            \;=\;\sum_{i=1}^{N_d} \boldsymbol{\tau}_{dist,i}(\mathbf{x},\mathrm{os}).

        Disturbance implementations may optionally accept a ``sat`` keyword argument; if so,
        the current :class:`~ADCS.satellite.Satellite` instance is provided.

        :param x: State vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Optional error/noise mode (typically used inside disturbances if supported).
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` | None

        :return: Disturbance torque vector in body frame, shape ``(3,)`` (N·m).
        :rtype: numpy.ndarray
        """
        torque_total = np.zeros(3)

        for idx, j in enumerate(self.disturbances):
            if self._dist_needs_sat[idx]:
                t = j.torque(sat=self, x=x, os=os)
            else:
                t = j.torque(x=x, os=os)

            torque_total += np.asarray(t).reshape(3,)

        return torque_total
    
    def act_torque(self, x: np.ndarray, u: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> np.ndarray:
        r"""
        Compute the total actuator torque.

        The total actuator torque is

        .. math::

            \boldsymbol{\tau}_{act}(\mathbf{x},\mathbf{u},\mathrm{os})
            \;=\;\sum_{j=1}^{N_u} \boldsymbol{\tau}_{act,j}(u_j,\mathbf{x},\mathrm{os}),

        where each actuator is a subclass of :class:`~ADCS.satellite_hardware.actuators.Actuator`.

        :param x: State vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Control input vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Error/noise mode passed to actuator torque models.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` | None

        :return: Actuator torque vector in body frame, shape ``(3,)`` (N·m).
        :rtype: numpy.ndarray
        """
        torque_total = np.zeros(3)
        
        # Use pre-computed indices to avoid repeated type checking
        for j in self._mtq_inds:
            torque_total = torque_total + self.actuators[j].torque(u[j], x, os=os, dmode=dmode)
        
        for j in self._rw_inds:
            torque_total = torque_total + self.actuators[j].torque(u[j], x, os=os, dmode=dmode)
        
        return torque_total

    
    def dist_torques_jacobian(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Union[np.ndarray, np.ndarray]:
        r"""
        Jacobians of total disturbance torque with respect to the state and disturbance parameters.

        Let :math:`\boldsymbol{\tau}_{dist}(\mathbf{x})` be the total disturbance torque. This method
        returns:

        .. math::

            \frac{\partial \boldsymbol{\tau}_{dist}}{\partial \mathbf{x}}
            \in \mathbb{R}^{n_x\times 3},
            \qquad
            \frac{\partial \boldsymbol{\tau}_{dist}}{\partial \boldsymbol{\theta}_d}
            \in \mathbb{R}^{n_{\theta_d}\times 3}.

        In the provided implementation, only the quaternion components affect the disturbance torque
        Jacobian, because environment vectors are rotated by attitude. Thus, the nonzero block is

        .. math::

            \left[\frac{\partial \boldsymbol{\tau}_{dist}}{\partial \mathbf{x}}\right]_{3:7,:}
            = \sum_i \frac{\partial \boldsymbol{\tau}_{dist,i}}{\partial \mathbf{q}}.

        Each disturbance model must implement:

        * ``torque_qjac(self, vecs)`` → :math:`\partial \boldsymbol{\tau}/\partial \mathbf{q}`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param vecs: Dictionary of body-frame environment vectors and their quaternion derivatives, e.g.
            ``{"b","r","s","v","rho","db","dr","ds","dv","os"}``.
        :type vecs: dict[str, numpy.ndarray]

        :return: Pair ``(ddist_torq__dx, ddist_torq__ddmp)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        .. rubric:: Return shapes

        +----------------------+--------------------------+
        | Name                 | Shape                    |
        +======================+==========================+
        | ``ddist_torq__dx``   | ``(state_len, 3)``       |
        +----------------------+--------------------------+
        | ``ddist_torq__ddmp`` | ``(dist_param_len, 3)``  |
        +----------------------+--------------------------+
        """
        ddist_torq__dx = np.zeros((self.state_len,3))
        ddist_torq__dx[3:7,:] = sum([j.torque_qjac(self,vecs) for j in self.disturbances],np.zeros((4,3)))
        ddist_torq__ddmp = np.zeros((0,3))
        return ddist_torq__dx,ddist_torq__ddmp

    def dist_torque_hess(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Hessian tensors of total disturbance torque.

        Returns second derivatives of :math:`\boldsymbol{\tau}_{dist}`:

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}_{dist}}{\partial \mathbf{x}^2},
            \qquad
            \frac{\partial^2 \boldsymbol{\tau}_{dist}}{\partial \mathbf{x}\,\partial \boldsymbol{\theta}_d},
            \qquad
            \frac{\partial^2 \boldsymbol{\tau}_{dist}}{\partial \boldsymbol{\theta}_d^2}.

        As with :meth:`~ADCS.satellite.Satellite.dist_torques_jacobian`, only quaternion components
        contribute in the current formulation, and each disturbance must implement:

        * ``torque_qqhess(self, vecs)`` → :math:`\partial^2 \boldsymbol{\tau}/\partial \mathbf{q}^2`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param vecs: Dictionary of body-frame environment vectors and their first/second quaternion derivatives.
        :type vecs: dict[str, numpy.ndarray]

        :return: Triple ``(dddist_torq__dxdx, dddist_torq__dxddmp, dddist_torq__ddmpddmp)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]

        .. rubric:: Return shapes

        +----------------------------+-----------------------------------------+
        | Name                       | Shape                                   |
        +============================+=========================================+
        | ``dddist_torq__dxdx``      | ``(state_len, state_len, 3)``           |
        +----------------------------+-----------------------------------------+
        | ``dddist_torq__dxddmp``    | ``(state_len, dist_param_len, 3)``      |
        +----------------------------+-----------------------------------------+
        | ``dddist_torq__ddmpddmp``  | ``(dist_param_len, dist_param_len, 3)`` |
        +----------------------------+-----------------------------------------+
        """
        dddist_torq__dxdx = np.zeros((self.state_len,self.state_len,3))
        dddist_torq__dxdx[3:7,3:7,:] = sum([j.torque_qqhess(self,vecs) for j in self.disturbances],np.zeros((4,4,3)))
        dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
        dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
        return dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp


    def dynJacCore(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> Union[np.ndarray, np.ndarray]:
        r"""
        Compute Jacobians of the dynamics with respect to state and input.

        This method returns the first-order linearization

        .. math::

            \delta \dot{\mathbf{x}}
            \approx
            \mathbf{A}\,\delta\mathbf{x} + \mathbf{B}\,\delta\mathbf{u},

        where

        .. math::

            \mathbf{A} = \frac{\partial f}{\partial \mathbf{x}},\qquad
            \mathbf{B} = \frac{\partial f}{\partial \mathbf{u}}.

        Environmental vectors (magnetic field, sun vector, velocity, position) are rotated into the body frame:

        .. math::

            \mathbf{v}_B(\mathbf{q}) = R(\mathbf{q})^\top \mathbf{v}_{ECI},

        and derivatives :math:`\partial \mathbf{v}_B/\partial \mathbf{q}` are obtained via
        :func:`~ADCS.helpers.math_helpers.drotmatTvecdq`.

        Actuator torque derivatives are aggregated via each actuator's methods, e.g.
        ``dtorq__dbasestate`` and ``dtorq__du``; disturbance derivatives are aggregated via
        :meth:`~ADCS.satellite.Satellite.dist_torques_jacobian`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Current control vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param orbital_state: Orbital/environmental state.
        :type orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List ``[dxdot__dx, dxdot__du]``.
        :rtype: list[numpy.ndarray]

        .. rubric:: Return shapes

        +--------------+-------------------------------+
        | Name         | Shape                         |
        +==============+===============================+
        | ``dxdot__dx``| ``(state_len, state_len)``    |
        +--------------+-------------------------------+
        | ``dxdot__du``| ``(control_len, state_len)``  |
        +--------------+-------------------------------+
        """
        R = orbital_state.R # Position in ECI [km]
        V = orbital_state.V # Velocity in body frame [km/s]
        B = orbital_state.B # Magnetic field in ECI [T]
        S = orbital_state.S # Sun Vector in ECI [km]
        rho = orbital_state.rho # Atmospheric density [kg/m^3]

        w = x[0:3]
        q = x[3:7]
        RWhs = x[7:]
        J = self.J_0
        invJ_noRW = self.invJ_noRW

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V

        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        vecs: Dict[str, Any] = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":orbital_state}
        com = self.COM

        ddist_torq__dx,ddist_torq__ddmp = self.dist_torques_jacobian(x,vecs)
        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],x,orbital_state) for j in range(len(self.actuators))],np.zeros((7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],x,orbital_state) for j in range(len(self.actuators))])

        dxdot__dx = np.zeros((self.state_len,self.state_len))
        dxdot__du = np.zeros((self.control_len,self.state_len))
        dxdot__dx[3,4:7] = 0.5*w
        dxdot__dx[4:7,3] = -0.5*w
        dxdot__dx[4:7,4:7] = 0.5*skewsym(w)
        dxdot__dx[0:3,3:7] = 0.5*Wmat(q).T
        dxdot__du[:,0:3] = dact_torq__du@invJ_noRW

        dxdot__dx[:,0:3] += ddist_torq__dx@invJ_noRW
        dxdot__dx[0:7,0:3] += dact_torq__dbase@invJ_noRW
        dxdot__dx[0:3,0:3] += (-skewsym(w@J)+J@skewsym(w))@invJ_noRW

        # Reaction Wheels
        if self.number_RW>0:
            dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],x,orbital_state) for j in range(len(self.actuators))])
            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            mRWjs = np.diagflat(RWjs)
            dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            dxdot__du[:,7:] = block_diag(*[self.actuators[j].dstor_torq__du(u[j],x,orbital_state) for j in range(len(self.actuators))])
            dxdot__du[:,7:] -= dxdot__du[:,0:3]@RWaxes.T@mRWjs
            dxdot__dx[0:7,7:] = np.hstack([self.actuators[j].dstor_torq__dbasestate(u[j],x,orbital_state) for j in range(len(self.actuators))])
            dxdot__dx[7:,7:] = np.diagflat([self.actuators[j].dstor_torq__dh(u[j],x,orbital_state) for j in self.momentum_inds])
            dxdot__dx[:,7:] -= dxdot__dx[:,0:3]@RWaxes.T@mRWjs
        return [dxdot__dx,dxdot__du]
    
    def dynamics_Hessians(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> List[List[np.ndarray]]:
        r"""
        Compute second-order partial derivatives (Hessians) of the dynamics.

        Returns the second derivatives of the dynamics map
        :math:`\dot{\mathbf{x}} = f(\mathbf{x},\mathbf{u},\mathrm{os})` with respect to state and input:

        .. math::

            \frac{\partial^2 f}{\partial \mathbf{x}^2},
            \qquad
            \frac{\partial^2 f}{\partial \mathbf{x}\partial \mathbf{u}},
            \qquad
            \frac{\partial^2 f}{\partial \mathbf{u}^2}.

        These are useful for second-order trajectory optimization methods (DDP/iLQR) and
        nonlinear uncertainty propagation.

        **Frame rotation derivatives**

        Environmental vectors are rotated into body frame and their first and second derivatives
        with respect to the quaternion are computed:

        .. math::

            \mathbf{v}_B(\mathbf{q}) = R(\mathbf{q})^\top \mathbf{v}_{ECI},\quad
            \frac{\partial \mathbf{v}_B}{\partial \mathbf{q}},\quad
            \frac{\partial^2 \mathbf{v}_B}{\partial \mathbf{q}^2},

        using :func:`~ADCS.helpers.math_helpers.drotmatTvecdq` and
        :func:`~ADCS.helpers.math_helpers.ddrotmatTvecdqdq`.

        **Output structure**

        The return is a nested list representing the block Hessians:

        .. code-block:: text

            [
              [ ddxdot__dxdx, ddxdot__dxdu ],
              [ ddxdot__dxdu.T, ddxdot__dudu ]
            ]

        where the final tensor index selects the component of :math:`\dot{\mathbf{x}}`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Current control vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param orbital_state: Orbital/environmental state.
        :type orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Nested list of Hessian tensors ``[[ddxdot__dxdx, ddxdot__dxdu],[ddxdot__dxdu.T, ddxdot__dudu]]``.
        :rtype: list[list[numpy.ndarray]]

        .. rubric:: Return shapes

        +-------------------+------------------------------------------+
        | Name              | Shape                                    |
        +===================+==========================================+
        | ``ddxdot__dxdx``  | ``(state_len, state_len, state_len)``    |
        +-------------------+------------------------------------------+
        | ``ddxdot__dxdu``  | ``(state_len, control_len, state_len)``  |
        +-------------------+------------------------------------------+
        | ``ddxdot__dudu``  | ``(control_len, control_len, state_len)``|
        +-------------------+------------------------------------------+
        """
        w = x[0:3]#.reshape((3,1))
        q = x[3:7]#normalize(x[3:7,:])
        RWhs = x[7:]
        invJ_noRW = self.invJ_noRW
        J = self.J

        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        ddR_B__dqdq = ddrotmatTvecdqdq(q,R)
        ddB_B__dqdq = ddrotmatTvecdqdq(q,B)
        ddV_B__dqdq = ddrotmatTvecdqdq(q,V)
        ddS_B__dqdq = ddrotmatTvecdqdq(q,S)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":orbital_state}
        com = self.COM

        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,3)))
        ddact_torq__dbasedbase = sum([self.actuators[j].ddtorq__dbasestatedbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
        ddact_torq__dudu = np.zeros((self.control_len,self.control_len,3))
        ddact_torq__dudbase = np.zeros((self.control_len,7,3))
        for j in range(len(self.actuators)):
            ddact_torq__dudu[j,j,:] = self.actuators[j].ddtorq__dudu(u[j],self,x,vecs)
            ddact_torq__dudbase[j,:,:] = self.actuators[j].ddtorq__dudbasestate(u[j],self,x,vecs)


        ddxdot__dxdx = np.zeros((self.state_len,self.state_len,self.state_len))
        ddxdot__dudu = np.zeros((self.control_len,self.control_len,self.state_len))
        ddxdot__dxdu = np.zeros((self.state_len,self.control_len,self.state_len))

        dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp = self.dist_torque_hess(x,vecs)

        ddxdot__dxdx[3,0:3,4:7]  = 0.5*np.eye(3)
        ddxdot__dxdx[4:7,0:3,3]  = 0.5*-np.eye(3)
        ddxdot__dxdx[4:7,0:3,4:7] = 0.5*-np.cross(np.expand_dims(np.eye(3),0),np.expand_dims(np.eye(3),1))
        ddxdot__dxdx[0:3,3:7,3:7] = np.transpose(ddxdot__dxdx[3:7,0:3,3:7],(1,0,2))

        ddxdot__dudu[:,:,0:3] = ddact_torq__dudu@invJ_noRW
        ddxdot__dxdu[0:7,:,0:3] = np.transpose(ddact_torq__dudbase,(1,0,2))@invJ_noRW
        ddxdot__dxdx[:,:,0:3] += dddist_torq__dxdx@invJ_noRW
        ddxdot__dxdx[0:7,0:7,0:3] += ddact_torq__dbasedbase@invJ_noRW
      
        JxI = np.cross(np.expand_dims(J,0),np.expand_dims(np.eye(3),1))
     
        ddxdot__dxdx[0:3,0:3,0:3] += (JxI + np.transpose( JxI,(1,0,2)))@invJ_noRW
        if self.number_RW>0:
            ddact_torq__dudh = np.zeros((self.control_len,self.number_RW,3))
            ddact_torq__dhdh = np.zeros((self.number_RW,self.number_RW,3))
            ddact_torq__dbasedh =  np.zeros((7,self.number_RW,3))
            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                ddact_torq__dudh[j,ind,:] = self.actuators[j].ddtorq__dudh(u[j],self,x,vecs)
                ddact_torq__dhdh[ind,ind,:] = self.actuators[j].ddtorq__dhdh(u[j],self,x,vecs)
                ddact_torq__dbasedh[:,ind,:] = np.squeeze(self.actuators[j].ddtorq__dbasestatedh(u[j],self,x,vecs))

            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])

            mRWjs = np.diagflat(RWjs)

            ddxdot__dxdu[7:,:,0:3] += np.transpose(ddact_torq__dudh,(1,0,2))@invJ_noRW
            ddxdot__dxdx[7:,0:7,0:3] += np.transpose(ddact_torq__dbasedh,(1,0,2))@invJ_noRW ###
            ddxdot__dxdx[0:7,7:,0:3] +=  ddact_torq__dbasedh@invJ_noRW


            AxI = -np.cross(np.expand_dims(RWaxes,1),np.expand_dims(np.eye(3),0))
            ddxdot__dxdx[7:,0:3,0:3] += -AxI@invJ_noRW
            ddxdot__dxdx[0:3,7:,0:3] += -np.transpose(AxI,(1,0,2))@invJ_noRW
            ddxdot__dxdx[7:,7:,0:3] += (ddact_torq__dhdh)@invJ_noRW

            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                ddxdot__dxdu[0:7,j,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dudbasestate(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdu[7+ind,j,7+ind] += np.transpose(self.actuators[j].ddstor_torq__dudh(u[j],self,x,vecs),(1,0,2))
                ddxdot__dudu[j,j,7+ind] = self.actuators[j].ddstor_torq__dudu(u[j],self,x,vecs)
                ddxdot__dxdx[0:7,0:7,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedbasestate(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,0:7,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdx[0:7,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dhdh(u[j],self,x,vecs))

            ddxdot__dxdu[:,:,7:] -= ddxdot__dxdu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dudu[:,:,7:] -= ddxdot__dudu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dxdx[:,:,7:] -= ddxdot__dxdx[:,:,0:3]@RWaxes.T@mRWjs

        return [[ddxdot__dxdx,ddxdot__dxdu],[ddxdot__dxdu.T,ddxdot__dudu]]


    def noiseless_rk4(self, x: np.ndarray, u: np.ndarray, dt: float, orbital_state0: Orbital_State, orbital_state1: Orbital_State, verbose: bool=False,mid_orbital_state: Orbital_State = None, quat_as_vec: bool = True, give_err_est = False) -> np.ndarray:
        r"""
        Propagate the state forward one step using RK4 (and optional embedded error estimate).

        Quaternion components are normalized at each substep to avoid drift.

        **RK4 update**

        .. math::

            \begin{aligned}
            \mathbf{k}_1 &= f(\mathbf{x}_n,\mathbf{u},\mathrm{os}_0) \\
            \mathbf{k}_2 &= f(\mathbf{x}_n+\tfrac{dt}{2}\mathbf{k}_1,\mathbf{u},\mathrm{os}_{1/2}) \\
            \mathbf{k}_3 &= f(\mathbf{x}_n+\tfrac{dt}{2}\mathbf{k}_2,\mathbf{u},\mathrm{os}_{1/2}) \\
            \mathbf{k}_4 &= f(\mathbf{x}_n+dt\,\mathbf{k}_3,\mathbf{u},\mathrm{os}_1) \\
            \mathbf{x}_{n+1} &= \mathbf{x}_n + \tfrac{dt}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)
            \end{aligned}

        where :math:`\mathrm{os}_{1/2}` is the midpoint orbital state; if not provided it is computed via
        :meth:`~ADCS.orbits.orbital_state.Orbital_State.average`.

        **Embedded error estimate (optional)**

        When ``give_err_est=True``, a third-order embedded estimate :math:`\mathbf{x}^{(3)}_{n+1}` is computed,
        and the elementwise difference is returned as a local truncation error proxy.

        **Noise/bias behavior**

        This routine enforces a deterministic propagation by using an
        :class:`~ADCS.satellite_hardware.errors.ErrorMode` with all noise/bias additions disabled.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param u: Control vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param dt: Integration time step (s).
        :type dt: float

        :param orbital_state0: Orbital state at the start of the step.
        :type orbital_state0: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param orbital_state1: Orbital state at the end of the step.
        :type orbital_state1: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param verbose: If ``True``, prints intermediate RK stages.
        :type verbose: bool

        :param mid_orbital_state: Optional precomputed midpoint orbital state.
        :type mid_orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State` | None

        :param quat_as_vec: If ``True``, propagate quaternions using RK4 on the quaternion components with normalization.
            If ``False``, use a commutator-free Lie group method (CG5) for quaternion propagation.
        :type quat_as_vec: bool

        :param give_err_est: If ``True``, return ``(x_next, err_est)`` using an embedded lower-order estimate.
        :type give_err_est: bool

        :return: Next state, or ``(next_state, error_estimate)`` if ``give_err_est=True``.
        :rtype: numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]

        :raises ValueError:
            If ``give_err_est=True`` while using the CG5 quaternion propagation variant.
        """
        x[3:7] = normalize(x[3:7])
        # Use no noise, no bias, no updates to either
        dmode = _NOISELESS_DMODE
        if quat_as_vec:
            if mid_orbital_state is None:
                mid_orbital_state = orbital_state0.average(orbital_state1)

            k1 = self.dynamics_core(x=x, u=u, orbital_state=orbital_state0, dmode=dmode, verbose=verbose)
            k2_in = x + 0.5 * dt * k1
            k2_in[3:7] = normalize(k2_in[3:7])
            k2 = self.dynamics_core(x=k2_in, u=u, orbital_state=mid_orbital_state, dmode=dmode, verbose=verbose)

            k3_in = x + 0.5 * dt * k2
            k3_in[3:7] = normalize(k3_in[3:7])
            k3 = self.dynamics_core(x=k3_in, u=u, orbital_state=mid_orbital_state, dmode=dmode, verbose=verbose)

            k4_in = x + dt * k3
            k4_in[3:7] = normalize(k4_in[3:7])
            k4 = self.dynamics_core(x=k4_in, u=u, orbital_state=orbital_state1, dmode=dmode, verbose=verbose,)

            out = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[3:7] = normalize(out[3:7])

            if verbose:
                print("k1:", k1)
                print("k2:", k2)
                print("k3:", k3)
                print("k4:", k4)

            if give_err_est:
                k33_in = x + dt * (2 * k2 - k1)
                k33_in[3:7] = normalize(k33_in[3:7])
                k33 = self.dynamics(k33_in, u, orbital_state1, dmode=dmode, verbose=verbose)

                out3 = x + (dt / 6.0) * (k1 + 4 * k2 + k33)
                out3[3:7] = normalize(out3[3:7])

                est_err = np.zeros(out.size - 1)
                est_err[:3] = np.abs(out[:3] - out3[:3])
                est_err[6:] = np.abs(out[7:] - out3[7:])
                est_err[3:6] = np.abs(quat_to_vec3(quat_mult(quat_inv(out[3:7]), out3[3:7]), 0))
                return out, est_err

            return out

        else:
            if give_err_est:
                raise ValueError("Error estimation not implemented for CG5 method.")

            ki = [np.zeros_like(x) for _ in range(5)]
            F = [np.zeros(3) for _ in range(5)]

            if mid_orbital_state is None:
                mid_orbital_state = [
                    orbital_state0.average(orbital_state1, CG5_c[i]) for i in range(5)
                ]

            for j in range(5):
                midstate = x + dt * sum([uc.CG5_a[j, i] * ki[i] for i in range(j)], np.zeros_like(x))
                if j > 0:
                    midstate[3:7] = normalize(
                        quat_mult(x[3:7], *[rot_exp(uc.CG5_a[j, i] * F[i]) for i in range(j)])
                    )
                ki[j] = self.dynamics_core(x=midstate, u=u, orbital_state=mid_orbital_state[j], dmode=dmode, verbose=verbose)
                F[j] = dt * midstate[0:3]

            out = x + dt * sum([uc.CG5_b[i] * ki[i] for i in range(5)], np.zeros_like(x))
            out[3:7] = normalize(
                quat_mult(x[3:7], *[rot_exp(uc.CG5_b[i] * F[i]) for i in range(5)])
            )

            return out
        
    

        
    def sensor_readings(self, x: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> np.ndarray:
        r"""
        Compute concatenated sensor readings (attitude sensors + wheel momentum measurements).

        Readings are collected from all non-GPS sensors in ``self.attitude_sensors`` and appended with
        reaction wheel momentum measurements from each :class:`~ADCS.satellite_hardware.actuators.RW`.

        Let :math:`\mathbf{y}` denote the stacked measurement vector:

        .. math::

            \mathbf{y} =
            \begin{bmatrix}
            \mathbf{y}_1 \\
            \vdots \\
            \mathbf{y}_{N_s} \\
            \hat{\mathbf{h}}_{RW}
            \end{bmatrix},

        where each :math:`\mathbf{y}_i` is a sensor-specific output (possibly vector-valued).

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Error/noise mode controlling sensor noise/bias injection.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` | None

        :return: Concatenated measurement vector.
        :rtype: numpy.ndarray
        """
        sensor_readings: List[np.ndarray] = [np.atleast_1d(self.attitude_sensors[j].reading(x=x, os=os, dmode=dmode)) for j in range(len(self.attitude_sensors))]
        rw_readings: List[np.ndarray] = [np.atleast_1d(self.rw_actuators[j].measure_momentum()) for j in range(len(self.rw_actuators))]
        combined_readings = sensor_readings + rw_readings
        if not combined_readings:
            return np.array([])
        return np.concatenate(sensor_readings + rw_readings)
    
    def noiseless_sensor_readings(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute sensor readings with noise disabled (bias may be included but not updated).

        Uses an :class:`~ADCS.satellite_hardware.errors.ErrorMode` configured as:

        * add_bias = True
        * add_noise = False
        * update_bias = False
        * update_noise = False

        This is useful for deterministic filtering tests where bias is treated as a fixed offset.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Concatenated measurement vector.
        :rtype: numpy.ndarray
        """
        dmode = ErrorMode(add_bias=True, add_noise=False, update_bias=False, update_noise=False)
        return self.sensor_readings(x=x, os=os, dmode=dmode)
    
    def noiseless_RW_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return noiseless reaction wheel momentum measurements.

        Delegates to each wheel's ``measure_momentum_noiseless()`` (implementation-specific).

        :param x: Current state vector (unused by some wheel models but included for interface consistency).
        :type x: numpy.ndarray

        :param os: Orbital/environmental state (unused by some wheel models but included for interface consistency).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of per-wheel measurement arrays.
        :rtype: list[numpy.ndarray]
        """
        return [rw.measure_momentum_noiseless() for rw in self.rw_actuators]
    
    def RW_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return reaction wheel momentum measurements (with wheel measurement model behavior).

        Delegates to each wheel's ``measure_momentum()``.

        :param x: Current state vector (unused by some wheel models but included for interface consistency).
        :type x: numpy.ndarray

        :param os: Orbital/environmental state (unused by some wheel models but included for interface consistency).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of per-wheel measurement arrays.
        :rtype: list[numpy.ndarray]
        """
        return [rw.measure_momentum() for rw in self.rw_actuators]


    def GPS_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return GPS sensor readings.

        GPS sensors are identified as instances of :class:`~ADCS.satellite_hardware.sensors.GPS`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of GPS reading arrays (model-specific shapes).
        :rtype: list[numpy.ndarray]
        """
        return [gps.reading(x=x, os=os) for gps in self.GPS_sensors]


    def gyro_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return readings from all gyroscopes.

        Filters ``self.attitude_sensors`` by :class:`~ADCS.satellite_hardware.sensors.Gyro`
        and returns ``sensor.reading(x, os)`` for each.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of gyro measurement arrays (typically angular rate, rad/s).
        :rtype: list[numpy.ndarray]
        """
        return [
            sensor.reading(x=x, os=os)
            for sensor in self.attitude_sensors
            if isinstance(sensor, Gyro)
        ]


    def mtm_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return readings from all magnetometers.

        Filters ``self.attitude_sensors`` by :class:`~ADCS.satellite_hardware.sensors.MTM`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of magnetometer measurement arrays (typically magnetic field, T).
        :rtype: list[numpy.ndarray]
        """
        return [
            sensor.reading(x=x, os=os)
            for sensor in self.attitude_sensors
            if isinstance(sensor, MTM)
        ]


    def sunpair_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return readings from all sun-pair sensors.

        Filters ``self.attitude_sensors`` by :class:`~ADCS.satellite_hardware.sensors.SunPair`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of sun-pair measurement arrays (model-specific shapes).
        :rtype: list[numpy.ndarray]
        """
        return [
            sensor.reading(x=x, os=os)
            for sensor in self.attitude_sensors
            if isinstance(sensor, SunPair)
        ]


    def sunsensor_readings(self, x: np.ndarray, os: Orbital_State) -> List[np.ndarray]:
        r"""
        Return readings from all sun sensors.

        Filters ``self.attitude_sensors`` by :class:`~ADCS.satellite_hardware.sensors.SunSensor`.

        :param x: Current state vector, shape ``(state_len,)``.
        :type x: numpy.ndarray

        :param os: Orbital/environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List of sun sensor measurement arrays (model-specific shapes).
        :rtype: list[numpy.ndarray]
        """
        return [
            sensor.reading(x=x, os=os)
            for sensor in self.attitude_sensors
            if isinstance(sensor, SunSensor)
        ]
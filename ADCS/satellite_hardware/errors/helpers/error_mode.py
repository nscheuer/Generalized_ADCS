__all__ = ["ErrorMode"]

from dataclasses import dataclass

@dataclass
class ErrorMode:
    r"""
    Disturbance Mode Configuration.

    This data class defines **binary control flags** that determine how a
    disturbance model contributes to spacecraft dynamics and how its internal
    stochastic components evolve over time.

    It is typically consumed by disturbance models derived from
    :class:`~ADCS.satellite_hardware.disturbances.disturbance.Disturbance`
    to control the inclusion of **bias terms** and **noise processes** in the
    applied disturbance torque or force.

    Mathematical Interpretation
    ----------------------------
    Let a generic disturbance be modeled as

    .. math::

        \mathbf{d}(t) = \mathbf{d}_0 + \mathbf{b}(t) + \mathbf{n}(t),

    where

    - :math:`\mathbf{d}_0` is the nominal deterministic disturbance,
    - :math:`\mathbf{b}(t)` is a bias term (typically slowly varying),
    - :math:`\mathbf{n}(t)` is a stochastic noise term.

    The boolean flags in this class determine whether each component is
    **included** in the dynamics and whether it is **time-varying**.

    Flag Semantics
    --------------
    :add_bias:
        If ``True``, the bias term :math:`\mathbf{b}(t)` is added to the
        disturbance dynamics.

    :add_noise:
        If ``True``, the noise term :math:`\mathbf{n}(t)` is added to the
        disturbance dynamics.

    :update_bias:
        If ``True``, the bias term evolves over time, typically according to
        a random walk model,

        .. math::

            \mathbf{b}_{k+1} = \mathbf{b}_k + \mathbf{w}_k,

        where :math:`\mathbf{w}_k` is a zero-mean process noise.

    :update_noise:
        If ``True``, a new noise realization is sampled at each evaluation step.

    :param add_bias: Enables inclusion of a bias term in the disturbance model.
    :type add_bias: bool
    :param add_noise: Enables inclusion of a stochastic noise term in the
        disturbance model.
    :type add_noise: bool
    :param update_bias: Enables time evolution of the bias term.
    :type update_bias: bool
    :param update_noise: Enables resampling of the noise term at each call.
    :type update_noise: bool
    :return: Disturbance mode configuration container.
    :rtype: DisturbanceMode
    """
    add_bias: bool               # Should the bias be added to dynamics?
    add_noise: bool              # Should noise be added to dynamics?
    update_bias: bool            # Should bias evolve (random walk)?
    update_noise: bool           # Should noise be resampled each call?
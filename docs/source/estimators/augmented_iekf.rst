Augmented Invariant Extended Kalman Filter
==========================================

This page describes the parameter estimation added by
:class:`~ADCS.estimators.attitude_estimators.attitude_AugmentedIEKF.AugmentedIEKF`.
It uses the invariant attitude error and measurement treatment described on
the :doc:`IEKF page <iekf>`, then adds Euclidean actuator-bias, sensor-bias,
and disturbance-parameter states.

Augmented state and error
-------------------------

The nominal state is

.. math::

   \hat x_a =
   (\hat{\boldsymbol\omega},\hat R,\hat{\mathbf h},
   \hat b_a,\hat b_s,\hat d).

The default attitude error is the right-invariant rotation vector
:math:`\operatorname{Log}(R\hat R^T)^\vee`. The parameter errors are additive:

.. math::

   \delta x_a=
   \begin{bmatrix}
   \delta\boldsymbol\omega &
   \delta\boldsymbol\theta &
   \delta\mathbf h &
   \delta b_a &
   \delta b_s &
   \delta d
   \end{bmatrix}^{T}.

For :math:`n_h` wheel states, :math:`n_a` actuator-bias components,
:math:`n_s` sensor-bias components, and :math:`n_d` disturbance parameters,
the covariance dimension is

.. math::

   n_P=6+n_h+n_a+n_s+n_d.

Construct the filter with an :class:`~ADCS.state.EstimatorState` whose
covariance uses that tangent dimension:

.. code-block:: python

   estimator = ADCS.AugmentedIEKF(
       estimated_satellite,
       initial_estimator_state,
       dt=dt,
       quaternion_order="left",
   )

``EstimatedSatellite`` determines the enabled parameter blocks from its
actuators, sensors, and disturbances. The state layout and covariance must
match those blocks.

1. Propagate the augmented nominal state
----------------------------------------

The estimated actuator biases and disturbance parameters are synchronized
to the estimated satellite before propagating the physical state. The
parameter means follow the configured random-walk model; in the deterministic
nominal propagation they remain constant unless the satellite model specifies
otherwise:

.. math::

   \hat x_{p,k+1}^-=f(\hat x_{p,k}^+,\hat b_{a,k}^+,\hat d_k^+,
   u_k,o_k,o_{k+1},\Delta t),
   \qquad
   \hat b_{s,k+1}^- = \hat b_{s,k}^+.

The propagation is the same physical spacecraft integration used by IEKF.

2. Linearize and discretize the augmented error dynamics
--------------------------------------------------------

The augmented continuous error system has the form

.. math::

   \delta\dot x_a=F_{a,k}\delta x_a+w_{a,k}.

The physical-state block uses the invariant tangent coordinates from the IEKF.
Columns for actuator biases and disturbance parameters enter the physical
rows through the satellite's dynamics Jacobians. Sensor biases do not drive
the spacecraft dynamics. The parameter rows are zero for constant nominal
parameters, while configured parameter random-walk rates populate the
continuous process-noise covariance:

.. math::

   F_a =
   \begin{bmatrix}
   F_{p,p} & F_{p,b_a} & 0 & F_{p,d}\\
   0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0
   \end{bmatrix},
   \qquad
   Q_{c,a}=\operatorname{blockdiag}(Q_{c,p},Q_{c,b_a},Q_{c,b_s},Q_{c,d}),

where the block row and column order is physical state, actuator bias, sensor
bias, and disturbance parameter. Van Loan discretization gives
:math:`\Phi_a` and :math:`Q_{d,a}^{VL}`. Held control noise and caller-supplied
discrete process covariance are added as in the base IEKF:

.. math::

   P_{a,k+1}^- = \Phi_aP_{a,k}^+\Phi_a^T+
   Q_{d,a}^{VL}+Q_{d,a}^{u}+Q_{d,a}^{state}.

3. Include parameter sensitivities in the measurement update
------------------------------------------------------------

The canonical measurement stack predicts sensor outputs using the current
physical estimate and estimated sensor biases. Its Jacobian contains the
physical invariant-error columns, sensor-bias columns, and any measurement
parameter sensitivities exposed by the sensor model:

.. math::

   r_k=z_k-\hat z_k,
   \qquad
   H_a=\begin{bmatrix}H_p & H_{b_a} & H_{b_s} & H_d\end{bmatrix}.

Most sensor channels have :math:`H_{b_a}=0` and :math:`H_d=0`; sensor-bias
columns are nonzero for channels whose biases are estimated. The IEKF's
invariant quaternion residual and inertial-frame direction residual
transformations apply to the physical attitude columns and to the complete
measurement row, including its bias sensitivities and covariance.

The gain and Joseph covariance update are

.. math::

   S=H_aP_a^-H_a^T+R,
   \qquad
   K=P_a^-H_a^TS^{-1},
   \qquad
   \delta x_a=Kr,

   P_{a,raw}^+=(I-KH_a)P_a^-(I-KH_a)^T+KRK^T.

The attitude block is retracted on the selected side of :math:`\hat R`; all
Euclidean parameter means receive their corresponding additive correction.
The full covariance, including cross-covariance between attitude and
parameters, is transported through the attitude reset Jacobian:

.. math::

   \hat R^+=\operatorname{Exp}(\delta\boldsymbol\theta^\wedge)\hat R^- ,
   \qquad
   \hat b_a^+=\hat b_a^-+\delta b_a,
   \quad
   \hat b_s^+=\hat b_s^-+\delta b_s,
   \quad
   \hat d^+=\hat d^-+\delta d,
   \qquad
   P_a^+=J_{reset}P_{a,raw}^+J_{reset}^T.

Initialization and covariance coordinates
------------------------------------------

The attitude determination routines report small-angle attitude uncertainty
in the body frame. For the default left-composed right-invariant error, warm
start rotates that attitude covariance into the inertial error frame before
assembling it with the additive parameter covariance blocks. User-supplied
``EstimatorState.cov`` and ``int_cov`` must already be expressed in the
selected invariant tangent coordinates and in the full augmented layout.

As for the base IEKF, augmented parameter coupling generally makes the
linearized error transition depend on the estimated trajectory. The augmented
filter preserves the invariant attitude error definition; it does not assume
that arbitrary torque and parameter models have autonomous group-affine error
dynamics.

Invariant Extended Kalman Filter (IEKF)
=======================================

This page follows the prediction and measurement update implemented by
:class:`~ADCS.estimators.attitude_estimators.attitude_IEKF.IEKF`. The default
configuration uses a **right-invariant attitude error**, with additive errors
for angular velocity and reaction-wheel momentum. The filter stores the
attitude in a unit quaternion but stores its uncertainty as a three-element
rotation vector.

The spacecraft-specific quaternion invariant EKFs were developed by Gui and
de Ruiter (2018), :doc:`listed in the citations <../contributing/citations>`.
This implementation uses their right-invariant attitude-error convention
with the package's general spacecraft tangent-linearized process model.

State and invariant error
-------------------------

Let :math:`R\in SO(3)` map body-frame vectors into the inertial frame. The
stored quaternion :math:`q` represents :math:`R`; the physical state is

.. math::

   x = (\boldsymbol\omega, R, \mathbf h),
   \qquad
   \boldsymbol\omega\in\mathbb R^3,\quad
   \mathbf h\in\mathbb R^{n_h}.

The default ``quaternion_order="left"`` composes attitude corrections on the
left. For the true attitude :math:`R` and estimate :math:`\hat R`, its
attitude error is

.. math::

   \eta_R = R\hat R^{T},
   \qquad
   \delta\boldsymbol\theta = \operatorname{Log}(\eta_R)^\vee.

This error is unchanged by a common right multiplication of both attitudes.
The full local error vector is

.. math::

   \delta x =
   \begin{bmatrix}
   \delta\boldsymbol\omega\\
   \delta\boldsymbol\theta\\
   \delta\mathbf h
   \end{bmatrix}
   \in\mathbb R^{6+n_h},
   \qquad
   \delta\boldsymbol\omega=\boldsymbol\omega-\hat{\boldsymbol\omega},
   \quad
   \delta\mathbf h=\mathbf h-\hat{\mathbf h}.

The attitude error is therefore three-dimensional even though the nominal
quaternion has four coefficients. In code, the default is available as:

.. code-block:: python

   estimator = ADCS.IEKF(
       estimated_satellite,
       initial_state,
       dt=dt,
       quaternion_order="left",  # right-invariant error R_true R_hat.T
   )

Setting ``quaternion_order="right"`` instead gives
:math:`\hat R^T R`, a left-invariant error. The IEKF uses the
``rotation_vector`` chart in either case.

1. Propagate the nominal spacecraft state
-----------------------------------------

The nominal estimate is propagated through the same deterministic spacecraft
model used by the simulation:

.. math::

   \hat x_{k+1}^{-} =
   f(\hat x_k^{+}, u_k, o_k, o_{k+1}, \Delta t).

The quaternion is integrated and normalized by the physical propagator. The
covariance remains in the local invariant-error coordinates. This step is
performed by :func:`~ADCS.estimators.process_model.propagate_state`.

2. Linearize the dynamics in invariant-error coordinates
--------------------------------------------------------

The continuous local error model is

.. math::

   \delta\dot x = F_k\delta x + w_k,
   \qquad
   \mathbb E[w_kw_k^T]=Q_{c,k}.

The spacecraft Jacobian is first formed in the stored state coordinates. Let
:math:`A_k` be that Jacobian and :math:`G(\hat q)` map the local tangent error
to a perturbation of the stored quaternion state. Because the tangent basis
moves with the nominal attitude, the local error matrix includes its rate:

.. math::

   F_k = G^\dagger(A_kG-\dot G),
   \qquad
   \delta x_{full}=G\,\delta x.

The code obtains :math:`A_k` from the satellite dynamics Jacobian and evaluates
:math:`\dot G` along the nominal quaternion trajectory. For the IEKF, both
maps use the selected ``quaternion_order`` and the rotation-vector chart.
Van Loan discretization produces the transition and continuous-model process
covariance:

.. math::

   \Phi_k = e^{F_k\Delta t},
   \qquad
   Q_{d,k}^{VL}=\int_0^{\Delta t}
   e^{F_k\tau}Q_{c,k}e^{F_k^T\tau}\,d\tau.

Held actuator-command noise adds the zero-order-hold term
:math:`Q_{d,k}^{u}=\Gamma_k Q_{u,k}\Gamma_k^T`. The caller-supplied discrete
``EstimatorState.int_cov`` is added separately. The covariance prediction is

.. math::

   P_{k+1}^{-}=\Phi_kP_k^{+}\Phi_k^T+
   Q_{d,k}^{VL}+Q_{d,k}^{u}+Q_{d,k}^{state}.

This is an invariant-error linearization of the configured spacecraft model.
General torques and disturbances can make :math:`F_k` depend on the nominal
trajectory; the group-affine, trajectory-independent error property is not
assumed here.

3. Select measurements and form innovations
--------------------------------------------

The measurement stack selects enabled, finite, and scheduled readings and
evaluates their predictions:

.. math::

   \hat z_k = h(\hat x_k^{-}, o_k).

Scalar and other additive channels use :math:`r=z-\hat z`. Quaternion attitude
measurements use the relative quaternion consistent with the attitude error:

.. math::

   \eta_{R,z}=R_z\hat R^{-T},
   \qquad
   r_q=\operatorname{Log}(\eta_{R,z})^\vee\in\mathbb R^3
   \quad (\texttt{quaternion_order="left"}).

Three-axis body-frame direction measurements from star trackers and Earth
horizon sensors have a natural invariant output error. Since
:math:`y=R^Tv` for an inertial reference vector :math:`v`, the residual is
expressed in the estimated inertial frame:

.. math::

   r_{inv}=\hat R(y-\hat y).

The same block rotation is applied to the corresponding measurement Jacobian
and covariance. Thus :math:`H` and :math:`R_k` use the same residual frame.
Other scalar sensors retain their native residual coordinates.

4. Build the measurement Jacobian and covariance
------------------------------------------------

The stack returns the active Jacobian in the selected local error coordinates
and projects quaternion measurement covariance into the three-element
rotation-vector residual. After the direction-output frame change, the update
uses

.. math::

   S_k=H_kP_k^-H_k^T+R_k,
   \qquad
   K_k=P_k^-H_k^TS_k^{-1},
   \qquad
   \delta x_k=K_kr_k.

5. Apply the correction and reset the covariance
------------------------------------------------

The non-attitude blocks are corrected additively. The attitude correction is
retracted on the left:

.. math::

   \hat R_k^+=\operatorname{Exp}(\delta\boldsymbol\theta_k^\wedge)\hat R_k^-.

In quaternion form this is
:math:`\hat q_k^+=\phi^{-1}(\delta\boldsymbol\theta_k)\otimes\hat q_k^-`.
The covariance uses the Joseph update followed by transport to the new local
error origin:

.. math::

   P_{raw}^+=(I-KH)P^-(I-KH)^T+KRK^T,
   \qquad
   P^+=J_{reset}P_{raw}^+J_{reset}^T.

The reset Jacobian is the rotation-vector chart Jacobian for the selected
composition order. These operations are owned by
:meth:`~ADCS.state.State.plus`,
:meth:`~ADCS.state.State.retraction_jacobian`, and
:meth:`~ADCS.state.State.transport_covariance`.

Initialization covariance
-------------------------

The attitude-determination routines report small-angle uncertainty in the
body frame. With the default right-invariant error, the warm-start attitude
block is rotated into the inertial error frame:

.. math::

   P_{\theta,left}=\hat R\,P_{\theta,body}\,\hat R^T.

An explicitly supplied ``EstimatorState.cov`` or ``int_cov`` must already
use the selected ``quaternion_order`` and rotation-vector coordinates.

Scope of the invariant property
-------------------------------

The attitude error and direction residual above use their stated group
actions. The spacecraft state also contains Euclidean rate and wheel-momentum
blocks, and arbitrary spacecraft torque models need not be group affine. The
filter therefore retains the correct model-dependent linearization and does
not claim autonomous error dynamics for all satellites. Estimated biases and
disturbance parameters are supported by
:class:`~ADCS.estimators.attitude_estimators.attitude_AugmentedIEKF.AugmentedIEKF`;
their equations are described on the
:doc:`augmented IEKF page <augmented_iekf>`.

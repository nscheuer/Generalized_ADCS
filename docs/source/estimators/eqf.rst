Equivariant Kalman Filter (EQF)
================================

:class:`~ADCS.estimators.attitude_estimators.attitude_EQF.EQF` estimates the
spacecraft estimator state on the product group
:math:`G=SO(3)\times\mathbb R^{3+n_h+n_a+n_s+n_d}`. A group element contains
the attitude :math:`R`, body angular velocity :math:`\omega`, reaction-wheel
momenta :math:`h`, estimated actuator and sensor biases, and disturbance
parameters. The stored attitude is a unit quaternion; its covariance uses a
three-element rotation-vector coordinate.

Group action and error
----------------------

The group product and the right-invariant error are

.. math::

   (R_1,z_1)(R_2,z_2)=(R_1R_2,z_1+z_2),\qquad
   E=X\hat X^{-1}=(R\hat R^T,z-\hat z),\quad
   z=(\omega,h,b_a,b_s,p_d).

The error coordinates at the identity are
:math:`\epsilon=(\delta\omega,\log(R\hat R^T)^\vee,\delta h,
\delta b_a,\delta b_s,\delta p_d)` in the package's state-block order. The
nominal attitude correction acts on the left and the other blocks are
additive.

Lift and prediction
-------------------

For the configured spacecraft dynamics :math:`\dot\omega=f_\omega(X,u,o)`
and :math:`\dot h=f_h(X,u,o)`, the right-trivialized lift is

.. math::

   \Lambda(X,u,o)=(f_\omega(X,u,o),\;R\omega,\;f_h(X,u,o),\;0,\;0,\;0).

The implementation evaluates this lift through the spacecraft's
``dynamics_core``, including actuator torque, environmental disturbance torque,
rigid-body coupling, reaction-wheel dynamics, and the effects of estimated
actuator biases and disturbance parameters. It then linearizes the
global group error at :math:`E=I`. Its attitude row has the exact kinematic
block :math:`A_{\theta\omega}=\hat R` and zero remaining attitude blocks.
The rate and wheel rows are central derivatives of the full nonlinear
dynamics under group perturbations. Each estimated parameter is matched to the
spacecraft model before its derivative is evaluated. Thus the complete
estimator state enters the EqF error generator :math:`A`; parameter rows are
zero for the deterministic random-walk model.

The nominal state is propagated with the shared nonlinear spacecraft
integrator. Continuous process noise is discretized with :math:`A` using Van
Loan's method. Held actuator-command noise and the state's configured
discrete process covariance are included. The covariance prediction is

.. math::

   P_{k+1}^{-}=\Phi_k P_k^{+}\Phi_k^T+Q_{d,k},\qquad
   \Phi_k=\exp(A_k\Delta t).

This is the general EqF construction for a system on a Lie group using an
equivariant extension of its input vector field. The code evaluates the
transformed error vector field directly, without constructing a basis for the
extended input space. As with any local Kalman approximation, the generator
is evaluated along the current estimate when configured torques are not group
affine; autonomous error dynamics are not assumed.

Measurements and correction
---------------------------

For a body-frame direction :math:`y=R^Tv` from a star tracker or Earth
horizon sensor, the equivariant residual and its Jacobian at the identity
error are

.. math::

   \hat y=\hat R^Tv,\qquad
   r=\hat R(y_{meas}-\hat y),\qquad
   H_\theta=[v]_\times.

Three active, independent single-axis magnetometers are assembled into one
body-frame magnetic vector and receive the same update. The direction noise
covariance is rotated into the same inertial frame. Quaternion attitude
measurements use the relative-rotation logarithm, and other sensors use their
local output Jacobians. A Joseph covariance update is followed by transport to
the new group-error origin.

.. code-block:: python

   estimator = ADCS.EQF(estimated_satellite, initial_state, dt=dt)
   estimate = estimator.predict(control, orbit_start, orbit_end)
   estimate = estimator.correct(measurements, orbit_end)

Scope
-----

This filter covers the full :class:`~ADCS.state.EstimatorState` layout. Its
estimated parameter blocks use additive group factors; their random-walk
process noise is retained. The physical spacecraft dynamics may be
non-equivariant under the original finite-dimensional inputs;
the general EqF input extension is implicit in the error-vector-field
calculation. The implementation does not assert a group-affine model for
arbitrary spacecraft torques.

The group-system construction follows `Mahony and Trumpf, “Equivariant Filter
Design for Kinematic Systems on Lie Groups”
<https://trumpf.id.au/pubs/Mahony_Trumpf_MTNS2020.pdf>`_. The output
linearization at the identity error follows `van Goor, Hamel, and Mahony,
“Equivariant Filter (EqF)” <https://doi.org/10.1109/TAC.2022.3194094>`_.

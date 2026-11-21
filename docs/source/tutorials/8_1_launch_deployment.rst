Launch and Deployment
=====================

Begin the mission timeline with the deployment interface requirements. A
launch provider may require a delayed activation or transmission window;
model that interval before enabling the relevant subsystem. A one-hour
power-off interval is an example, not a universal CubeSat rule.

For instance, set a mission parameter such as ``deployment_wait_s = 3600``
and start detumbling after that interval if the provider requires the
spacecraft to remain off. Model radio transmission separately when power-on
and radio-inhibit times differ. The `CubeSat Design Specification
<https://www.nasa.gov/wp-content/uploads/2018/01/cubesatdesignspecificationrev14_12022-02-09.pdf>`__
gives an example requiring at least 45 minutes before transmission while
allowing power-on immediately after deployment. Use the requirements from
your own dispenser and launch provider.

.. admonition:: Mission design tips and tricks
   :class: mission-recommendations

   **Hardware layout**

   * Align each reaction wheel's spin axis so its line of action passes
     through the spacecraft center of mass. Place magnetorquers near the
     center of mass.
   * Keep magnetometers away from magnetorquers, high-current electronics,
     and their wiring. Schedule magnetometer readings when torquers are off
     or their magnetic interference has settled. Use quiet measurement
     windows and account for the settling time after a torquer pulse.
   * Estimate the available slew-rate scale from wheel momentum margin and
     the largest principal moment of inertia:
     :math:`\omega_{\max} \lesssim h_{\mathrm{available}}/J_{\max}`.
     Wheel torque, axis geometry, and disturbances impose further limits.

   **Models and validation**

   * Select actuators and sensors using controllability and observability
     analysis, then check mission-level pointing, power, and pass performance
     in simulation.
   * Measure sensor and actuator noise, bias, and drift on the ground. Include
     these measured properties in the simulation and estimator.
   * Carry inertia uncertainty in the estimation loop and test it in Monte
     Carlo runs. :doc:`Monte Carlo <7_0_monte_carlo>` shows how to sample
     operating conditions and compare trajectories.
   * Add sensor redundancy where mass, power, and interfaces allow it. Plan
     an independent backup communications path, such as a `Globalstar service
     <https://www.globalstar.com/en-us/about/our-technology>`__, if its
     coverage, link budget, licensing, and integration fit the mission.
   * Check pointing error across the full visible ground pass, not only at
     its final sample. Track wheel momentum and torque margins throughout.

The first active ADCS phase is :doc:`Detumbling <8_2_detumbling>`.

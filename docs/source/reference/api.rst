API Reference
=============

The API reference is generated from the docstrings in the ``ADCS`` package.
This page groups the subpackages by task; :doc:`../ADCS` lists every module.
For how the pieces fit together, see :doc:`../getting_started/concepts`.

Run a simulation
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: ``ADCS.simulate``
      :link: ../ADCS.simulate
      :link-type: doc

      ``simulate()``: integrate the dynamics, simulate sensors, and run the
      estimators, goal and controller.

   .. grid-item-card:: ``ADCS.state``
      :link: ../ADCS.state
      :link-type: doc

      ``State`` and ``EstimatorState``, the named spacecraft and estimator
      states.

   .. grid-item-card:: ``ADCS.mc``
      :link: ../ADCS.mc
      :link-type: doc

      ``simulate_mc`` and ``MCConfig`` for Monte Carlo ensembles.

   .. grid-item-card:: ``ADCS.simulate_remote`` · ``ADCS.remote``
      :link: ../ADCS.simulate_remote
      :link-type: doc

      Run the controller or estimators on another computer (see also
      :doc:`../ADCS.remote`).

Model the spacecraft and its environment
----------------------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: ``ADCS.satellite_hardware``
      :link: ../ADCS.satellite_hardware
      :link-type: doc

      ``Satellite``, actuators, sensors, disturbances, and noise and bias
      models.

   .. grid-item-card:: ``ADCS.satellite_factory``
      :link: ../ADCS.satellite_factory
      :link-type: doc

      Preset spacecraft and hardware; see also the
      :doc:`../satellite_factory/index` catalogue.

   .. grid-item-card:: ``ADCS.orbits``
      :link: ../ADCS.orbits
      :link-type: doc

      ``Orbital_State`` and orbit propagation (Keplerian, SGP4).

   .. grid-item-card:: ``ADCS.environment``
      :link: ../ADCS.environment
      :link-type: doc

      ``Environment``: magnetic field, atmosphere, ephemeris and star
      catalogue models.

Estimate and control
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: ``ADCS.estimators``
      :link: ../ADCS.estimators
      :link-type: doc

      Attitude determination, attitude estimators (EKF, MEKF, UKF, SRUKF and
      augmented variants) and orbit estimators.

   .. grid-item-card:: ``ADCS.controller``
      :link: ../ADCS.controller
      :link-type: doc

      The ``Controller`` base class, magnetic and wheel controllers, LP/QP
      allocation, plan-and-track and SALTRO.

   .. grid-item-card:: ``ADCS.CONOPS``
      :link: ../ADCS.CONOPS
      :link-type: doc

      Pointing goals and ``GoalList`` schedules.

   .. grid-item-card:: ``ADCS.flight_software``
      :link: ../ADCS.flight_software
      :link-type: doc

      Flight-software modules: single-core execution and tasks.

Analyse and visualize
---------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: ``ADCS.helpers``
      :link: ../ADCS.helpers
      :link-type: doc

      ``SimulationResults``, plotting (``ADCS.plot`` and ``ADCS.plots``), the
      Godot viewer and math helpers.

   .. grid-item-card:: ``ADCS.examples``
      :link: ../ADCS.examples
      :link-type: doc

      The small examples bundled with the package.

.. toctree::
   :hidden:

   ../ADCS

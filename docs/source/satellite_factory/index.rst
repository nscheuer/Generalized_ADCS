Satellite Factory
=================

These pages summarize the preset spacecraft and hardware factories available in
``ADCS.satellite_factory``.

Factory source links are listed on each spacecraft and component table next to
the affected preset. These sources document preset data values such as hardware
configuration, inertia, center of mass, sensor errors, actuator authority, and
mission-specific assumptions. Package-level algorithm and model references are
kept separately in :doc:`../contributing/citations`.

Using a preset
--------------

Each preset is a ``create_*`` function. Spacecraft presets return a complete
``ADCS.Satellite`` with actuators, sensors and disturbance models; hardware
presets return the components, ready to pass to ``ADCS.Satellite``:

.. code-block:: python

   from ADCS.satellite_factory import create_beavercube2_cubesat
   from ADCS.satellite_factory.actuators import create_sfl_reaction_wheels

   satellite = create_beavercube2_cubesat()   # 4 actuators, 8 sensors
   wheels = create_sfl_reaction_wheels()      # one RW per axis

Pass ``estimated=True`` to a spacecraft preset to get an
``EstimatedSatellite``, the model used by an estimator. The tables on each page
are sortable: click the **Tier** or **Model** header.

Catalogue
---------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Satellites
      :link: satellites
      :link-type: doc
      :footer: 9 presets

      Complete spacecraft, from the BeaverCube missions to LightSail 2.

   .. grid-item-card:: Reaction wheels
      :link: reaction_wheels
      :link-type: doc
      :footer: 3 presets

      Wheel layouts, authority limits and inertia.

   .. grid-item-card:: Magnetorquers
      :link: magnetorquers
      :link-type: doc
      :footer: 5 presets

      Torque rods, air-core coils and PCB magnetorquers.

   .. grid-item-card:: Magnetometers
      :link: magnetometers
      :link-type: doc
      :footer: 7 presets

      Three-axis magnetometers.

   .. grid-item-card:: Gyroscopes
      :link: gyroscopes
      :link-type: doc
      :footer: 7 presets

      Rate sensors and IMUs.

   .. grid-item-card:: Sun sensors
      :link: sun_sensors
      :link-type: doc
      :footer: 6 presets

      Coarse sun sensors and solar-panel arrays.

   .. grid-item-card:: Star trackers
      :link: star_trackers
      :link-type: doc
      :footer: 6 presets

      Vector and quaternion star-tracker models.

   .. grid-item-card:: GPS
      :link: gps
      :link-type: doc
      :footer: 1 preset

      GPS receivers.

   .. grid-item-card:: Earth horizon sensors
      :link: earth_horizon_sensors
      :link-type: doc
      :footer: 2 presets

      Horizon sensors.

.. toctree::
   :hidden:
   :maxdepth: 1

   satellites
   reaction_wheels
   magnetorquers
   magnetometers
   gyroscopes
   sun_sensors
   star_trackers
   gps
   earth_horizon_sensors

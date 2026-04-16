0.1.1 Earth Horizon Sensor Added (2026-04-15)
=============================================

Earth horizon sensing is now available in Generalized_ADCS through
``EarthHorizonSensor``. This sensor models a horizon detector that returns a
body-frame nadir/horizon direction measurement when Earth is inside the sensor
field of view, with configurable boresight and noise behavior. In practice, it
provides an additional attitude reference that complements sun and magnetic
measurements, and can be used in simulation and estimator pipelines to improve
robustness for nadir-pointing or low-observability conditions.

- API page: :doc:`EarthHorizonSensor <../ADCS.satellite_hardware.sensors.earth_horizon>`
- Direct docs link: `https://nscheuer.github.io/Generalized_ADCS/ADCS.satellite_hardware.sensors.earth_horizon.html <https://nscheuer.github.io/Generalized_ADCS/ADCS.satellite_hardware.sensors.earth_horizon.html>`_

.. image:: ../_static/release_notes/0_1_1_earth_horizon.png
   :alt: Earth horizon image for release note 0.1.1.
   :width: 700px
   :align: center

Image source: `NASA - Sun shines above Earth's horizon <https://www.nasa.gov/image-article/sun-shines-above-earths-horizon/>`_.
Filter Justification
====================

Can you determine the satellite's orientation from these raw measurements?
Six coarse Sun sensors face the positive and negative body axes. A gyroscope
and magnetometer measure along each body axis. Each trace shows one raw
measurement over time. The readings include sensor noise, and the gyro also
has a fixed bias. Try to work out which way the satellite is pointing, and
how it moves, using only these measurements.

.. image:: ../_static/tutorials/estimators/filter_justification.png
   :alt: Time series of six Sun sensor, three gyroscope, and three magnetometer measurements.
   :width: 100%

Run the short script below to generate the plot. Does every possible
orientation produce a different set of readings?

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/estimators/filter_justification.py

.. literalinclude:: ../../../examples/tutorials/estimators/filter_justification.py
   :language: python
   :caption: examples/tutorials/estimators/filter_justification.py

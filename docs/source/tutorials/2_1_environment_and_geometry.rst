Environment and Geometry
========================

Environment
-----------

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/torques_on_a_spacecraft/environment_and_geometry.py

.. literalinclude:: ../../../examples/tutorials/torques_on_a_spacecraft/environment_and_geometry.py
   :language: python
   :caption: examples/tutorials/torques_on_a_spacecraft/environment_and_geometry.py
   :start-at: import ADCS
   :end-before: size =

Geometry
--------

A 10 × 30 × 10 cm 3U body; body *y* is the long axis.

.. literalinclude:: ../../../examples/tutorials/torques_on_a_spacecraft/environment_and_geometry.py
   :language: python
   :start-at: size =

Wheel Momentum Accumulation
===========================

.. image:: ../_static/tutorials/controlling_spacecraft/wheel_momentum_accumulation.gif
   :alt: A three-wheel spacecraft holds nadir pointing while orbiting Earth.
   :width: 100%

Three orthogonal reaction wheels hold nadir pointing through drag, solar
radiation pressure, gravity-gradient torque, and residual magnetic torque.
The wheels accumulate momentum because this controller does not unload it.

.. image:: ../_static/tutorials/controlling_spacecraft/wheel_momentum_accumulation.png
   :alt: ADCS.plot shows the three wheel momenta and their increasing combined magnitude.
   :width: 100%

The black line is the magnitude of the combined wheel momentum.

.. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/wheel_momentum_accumulation.py
   :language: python
   :caption: wheel_momentum_accumulation.py

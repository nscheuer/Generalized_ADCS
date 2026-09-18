0.1.8 Available on PyPI (2026-08-19)
====================================

Generalized ADCS is now available on PyPI:

https://pypi.org/project/Generalized-ADCS/

Install it with pip::

    pip install generalized-adcs

The package can then be imported like any other Python dependency::

    import ADCS
    import numpy as np

Generalized control allocation
------------------------------

This release adds generalized control allocation for spacecraft with different
actuator combinations and constraints. The allocation interfaces support
mapping requested control effort to available actuators, including mixed
magnetorquer and reaction-wheel systems.

The control-allocation methods and their application to small-spacecraft
attitude control are described in the :doc:`SSC26 paper <../ssc26/paper>`.

State class
-----------

Spacecraft states are represented by ``ADCS.State`` objects containing angular
velocity ``w``, attitude quaternion ``q``, and reaction-wheel momentum ``h``.
Use explicit conversions at numerical-library boundaries::

    import ADCS
    import numpy as np

    # Established ordering: [w(3), q(4), h(n_rw)]
    state = ADCS.State.from_array(
        np.array([0.01, -0.02, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0])
    )

    print(state.w)                 # angular velocity
    print(state.q)                 # body-to-ECI quaternion
    print(state.h)                 # reaction-wheel momentum
    state_vector = state.as_array() # convert back to a NumPy array

The class also provides quaternion-aware state operations, including normalized
copies and manifold-consistent error calculations::

    normalized_state = state.normalized()
    error = normalized_state.subtract(state)
    reconstructed_state = state.add_error(error)

Bug fixes
---------

- Added missing runtime dependencies to the package metadata.
- Replaced a non-PyPI dependency with an in-tree Cholesky update helper.
- Corrected public type annotations for downstream type checkers.
- Relaxed runtime dependency pins to improve compatibility with existing
  environments.

This release also ships package metadata for Python type checkers, keeps
runtime dependencies installable from standard PyPI packages, and includes
general bug fixes and compatibility improvements across the simulation,
control, and packaging interfaces.

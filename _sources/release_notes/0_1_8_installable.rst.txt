0.1.8 Installable from PyPI (2026-08-05)
========================================

``pip install generalized-adcs`` now works, and the package it installs can
actually be imported.

Two undeclared hard dependencies
--------------------------------

``pyproject.toml`` declared neither ``rich`` (imported by
``ADCS.mc.monte_carlo_runner``, which ``ADCS/__init__.py`` reaches) nor
``choldate`` (imported by the square-root UKF). Both were hard, top-level
imports, so a normal install produced a package that raised ``ImportError`` on
``import ADCS``. Continuous integration never caught it because CI installs
from ``requirements.txt``, which *did* list them — the two files had drifted
apart.

The Cholesky update is now in-tree
----------------------------------

``choldate`` has no PyPI release at all. It installs only from git, needs
``--no-build-isolation``, and — being a direct URL reference — made this
distribution impossible to publish, since PyPI rejects such metadata outright.

:mod:`ADCS.helpers.cholesky_update` now implements the rank-1 Cholesky update
and downdate directly, as the standard LINPACK sequence of Givens rotations,
``njit``-compiled. It matches ``choldate`` to roughly ``1e-15`` and is covered
by 51 tests.

Two deliberate improvements over the package it replaces:

* The caller's update vector is no longer destroyed. ``choldate`` used it as
  scratch space.
* A downdate whose result is not positive definite now yields ``NaN``.
  ``choldate`` neither raised nor returned ``NaN`` — it computed
  :math:`\sqrt{|r^2|}` and returned a plausible-looking but incorrect factor.
  The SRUAKF has always guarded on ``np.any(np.isnan(...))``; that guard could
  never fire before, and now does.

Dependencies are ranges, not pins
---------------------------------

Runtime dependencies are declared as compatible ranges rather than exact
patch pins, so the package installs into an existing environment instead of
fighting it. Transitive dependencies are left to pip. Build- and test-only
packages moved to extras: ``[cpp]`` for the C++ add-on toolchain, ``[dev]``
for the test tooling, ``[viz]`` for the 3-D orbit animation helpers.

The junk placeholder ``np==1.0.2`` — a squatter package on PyPI, not NumPy,
which itself fails under NumPy 2.x — has been removed.

For deterministic reproduction of published campaign results, the exact
versions are preserved in ``requirements-repro.txt``::

    pip install -r requirements-repro.txt
    pip install -e . --no-deps

.. note::

   The version jumps from the previously released 0.1.5 to 0.1.8 because
   ``pyproject.toml`` had not been bumped alongside the 0.1.6 and 0.1.7
   release notes.

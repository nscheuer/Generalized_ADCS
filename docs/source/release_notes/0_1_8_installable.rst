0.1.8 Available on PyPI (2026-08-19)
====================================

Generalized ADCS is now available on PyPI:

https://pypi.org/project/Generalized-ADCS/

Install it with pip::

    pip install generalized-adcs

Bug-fixes:

- Added missing runtime dependencies to the package metadata.
- Replaced a non-PyPI dependency with an in-tree Cholesky update helper.
- Corrected public type annotations for downstream type checkers.
- Relaxed runtime dependency pins to improve compatibility with existing
  environments.

This release also ships package metadata for Python type checkers and keeps
runtime dependencies installable from standard PyPI packages.

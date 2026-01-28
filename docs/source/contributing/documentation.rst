Documentation Guidelines
========================

Overview
--------

The Generalized ADCS documentation is **auto-generated from Python class
and function docstrings** using Sphinx.

After building, the documentation can be viewed locally by opening the
generated ``index.html`` file in the build directory.

Building the Documentation
--------------------------

Ensure that the virtual environment is activated, then rebuild the
documentation from the ``docs`` directory:

.. code-block:: bash

   source venv/bin/activate
   cd docs/
   make clean
   sphinx-build -W -b html source build

The ``-W`` flag treats all warnings as errors. **The build must complete
without reStructuredText errors.**

Viewing the Documentation
-------------------------

Open the generated documentation locally:

``docs/build/html/index.html``

Registering New Modules
-----------------------

When adding new Python modules, regenerate the API documentation:

.. code-block:: bash

   cd docs/
   sphinx-apidoc -f -e --maxdepth 1 -o source/ ../ADCS/

All new public functions and classes **must** be included in the module-level
``__all__`` declaration:

.. code-block:: python

   __all__ = ["MyClass", "my_function"]

AI-Assisted Documentation
-------------------------
You can use the following AI prompt to help generate docstrings for new functions:

.. code-block:: text

    Write a Sphinx+LaTeX documentation for these classes and functions, including:
    - r""" strings
    - Full math explanation with equations
    - When referencing other classes or methods, use the full link :class:`~ADCS.etc`
    - STRICTLY use :param:, :type:, :return:, :rtype:
    - When using or creating tables, follow reST rules, means no ** allowed
    - Every docstring field list must end with a blank line.
    Write only the function header and the documentation.

    Review your results and compare it with the rules above. If one of the rules is broken, notice and fix it.

Example
-------

.. code-block:: python

   def quat_diff(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
       r"""
       Compute the relative quaternion between two attitudes.

       The quaternion error is defined as

       .. math::

           \mathbf{q}_{\mathrm{err}} =
           \mathbf{q}_0^{-1} \otimes \mathbf{q}_1

       The result is normalized and forced to have a non-negative scalar part,
       ensuring a unique minimal-angle representation.

       :param q0: Reference quaternion.
       :type q0: numpy.ndarray
       :param q1: Target quaternion.
       :type q1: numpy.ndarray
       :return: Relative quaternion mapping ``q0`` to ``q1``.
       :rtype: numpy.ndarray

       """

Testing Guidelines
==================

All new functions, classes, or modules **must include corresponding tests**
in the ``testing/`` directory.

Running Tests
-------------

To run the full test suite (as executed by GitHub Actions):

.. code-block:: bash

   cd testing/
   pytest -v

Note that running all tests may take several minutes.

Recommended Local Testing
-------------------------

When developing locally, it is recommended to skip very slow tests:

.. code-block:: bash

   cd testing/
   pytest -k "not vslow"

This significantly reduces runtime while still covering most functionality.

Testing Individual Files
------------------------

To run a specific test file:

.. code-block:: bash

   cd testing/
   pytest ./test_controllers/test_controller_mtq_w_rw.py

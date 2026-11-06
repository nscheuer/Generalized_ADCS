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

Local-only Tests
----------------

The Godot viewer test is marked ``local`` because it requires a local Godot
binary and viewer source tree. To run the regular Python suite without
local-only tests:

.. code-block:: bash

   pytest -m "not local" testing/

To run only the Godot test:

.. code-block:: bash

   pytest -m local testing/test_viz/test_viewer_godot.py

Testing Individual Files
------------------------

To run a specific test file:

.. code-block:: bash

   cd testing/
   pytest ./test_controllers/test_controller_mtq_w_rw.py

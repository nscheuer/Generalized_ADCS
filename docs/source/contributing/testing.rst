Testing Guidelines
==================

All new functions, classes, or modules **must include corresponding tests**
in the ``testing/`` directory.

Running Tests
-------------

To run the full test suite, run ``pytest`` from the repository root with the
development tools installed (``pip install -e ".[dev]"`` or
``pip install -r requirements.txt``). GitHub Actions runs the suite this way
on every push and pull request:

.. code-block:: bash

   pytest testing/ -m "not local"

Add ``-v`` to list each test as it runs. Note that running all tests may take
several minutes (about 3 minutes in the earlier testing notes).

The pytest settings in ``pyproject.toml`` already set ``testing/`` as the test
path and exclude ``local`` tests, so a plain ``pytest`` from the repository
root runs the same selection.

Local-only Tests
----------------

The Godot viewer test is marked ``local`` because it requires a local Godot
binary and viewer source tree. Local tests are excluded by default (see
above); to run the regular Python suite explicitly without them:

.. code-block:: bash

   pytest -m "not local" testing/

To run only the Godot test:

.. code-block:: bash

   pytest -m local testing/test_viz/test_viewer_godot.py

Testing Individual Files
------------------------

To run a specific test file, pass its path:

.. code-block:: bash

   pytest testing/test_controllers/mtq_w_rw/test_controller_mtq_w_rw.py

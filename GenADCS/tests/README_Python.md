# Python tests

**Build.** From the repository root, activate the project environment with `source venv/bin/activate` and install dependencies with `python -m pip install -r requirements.txt`. The Python backend needs no compilation.

**Run tests.** Run `python -m pytest GenADCS/tests/backends/python` from the repository root. The binding, conformance, and differential suites also exercise Python APIs, but require the C++ and Rust extensions to be built first.

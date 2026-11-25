# GenADCS development commands

Run these commands from the repository root. The package requires Python 3.10 or newer. The system package commands below target Debian or Ubuntu.

| Python | C++ | Rust |
| --- | --- | --- |
| **Minimum setup to run `GenADCS/`**<br>`sudo apt install python3-venv`<br>`python3 -m venv .venv`<br>`source .venv/bin/activate`<br>`python -m pip install -r requirements.txt`<br>`just -f just/build.python.just` | **Minimum setup to run `GenADCS/`**<br>Complete the Python setup, then:<br>`sudo apt install build-essential clang cmake ninja-build libeigen3-dev sccache`<br>`just -f just/build.cpp.just` | **Minimum setup to run `GenADCS/`**<br>Complete the Python setup, install Rust and Cargo with rustup, then:<br>`sudo apt install build-essential sccache`<br>`just -f just/build.rust.just` |
| **Additional checks**<br>When `strict.python.just` is added:<br>`just -f just/strict.python.just` | **Additional checks**<br>`sudo apt install clang-tidy llvm`<br>Install `osv-scanner` for the `audit` target.<br>`just -f just/strict.cpp.just`<br>The default recipe runs `lint`, `analyze`, `ubsan`, `asan`, `tsan`, `coverage`, and `audit`. | **Additional checks**<br>When `strict.rust.just` is added:<br>`just -f just/strict.rust.just` |

The Python and Rust strict recipes are planned; only `strict.cpp.just` exists currently. Activate `.venv` before running any recipe so CMake and PyO3 use its Python interpreter.

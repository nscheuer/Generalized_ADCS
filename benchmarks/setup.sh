#!/usr/bin/env bash
# Create one virtual environment per benchmarked tool under .bench/ and build
# the Generalized ADCS 3.0 Python, C++ and Rust backends with the project's own
# just recipes. Run from the repository root:
#
#     bash benchmarks/setup.sh
#
# Needs: python3 (>= 3.10) with venv, git, and for the 3.0 native backends the
# toolchain the build recipes expect:
#     sudo apt install clang ninja-build libeigen3-dev sccache
#     rustup (cargo), cargo install just
# Basilisk and Orekit install from pip wheels; Orekit's Java runtime comes from
# the jdk4py wheel, so no system Java is needed.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
BENCH=.bench
mkdir -p "$BENCH"

for tool in python3 cargo just ninja sccache cmake; do
    command -v "$tool" >/dev/null || { echo "missing: $tool (see the header of this script)"; exit 1; }
done

new_venv() {
    local name=$1
    if [[ ! -x "$BENCH/$name/bin/python" ]]; then
        python3 -m venv "$BENCH/$name"
        "$BENCH/$name/bin/python" -m pip install -q --upgrade pip
    fi
}

echo "==> Generalized ADCS 3.0 (Python, C++, Rust backends)"
new_venv genadcs3
(
    # The just recipes use the active environment's python.
    source "$BENCH/genadcs3/bin/activate"
    just -f build.python.just build
    just -f build.cpp.just build
    just -f build.rust.just build
)

echo "==> Generalized ADCS 2.x (PyPI release)"
new_venv adcs2
"$BENCH/adcs2/bin/pip" install -q "generalized-adcs==0.2.2"

echo "==> Basilisk"
new_venv basilisk
"$BENCH/basilisk/bin/pip" install -q "bsk==2.12.2"

echo "==> Orekit"
new_venv orekit
"$BENCH/orekit/bin/pip" install -q "orekit-jpype==13.1.9" jdk4py numpy

echo
echo "Done. Run the suite with:  python3 benchmarks/run_all.py"

"""The SSC26 poster's printed code must keep running.

``papers/SSC26_poster/verify_snippets.py`` holds every code block printed on
poster SSC26-P2-54 verbatim, between ``--- PANEL x ---`` markers, and executes
it. That makes it the machine-checkable contract between a printed artifact we
cannot revise and the package we keep changing: if an API moves, the poster is
wrong in public, and this is what says so.

It was written as a script and never wired into CI, so the claim on
``/ssc26`` -- and in the Colab notebook -- that the snippets are executed
automatically was aspirational. This test makes it true at PR time; the
``smoke-test`` job in ``.github/workflows/publish.yml`` runs the same script
against the built wheel at release time, which is the stronger check because it
exercises what a user actually receives.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = REPO_ROOT / "papers" / "SSC26_poster" / "verify_snippets.py"


def test_verify_snippets_script_exists():
    """A rename that orphans the script must fail loudly, not skip silently."""
    assert SNIPPETS.is_file(), (
        f"{SNIPPETS.relative_to(REPO_ROOT)} is missing. The poster's printed "
        "code is no longer verified by anything -- restore it or update this "
        "test and the claims on docs/source/ssc26/."
    )


def test_poster_snippets_still_run():
    """Every code block printed on the poster executes against this tree."""
    proc = subprocess.run(
        [sys.executable, str(SNIPPETS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "Poster snippets no longer run against this tree. Poster "
            "SSC26-P2-54 is printed and cannot be revised, so either the API "
            "change is wrong or the landing page must stop claiming the "
            f"printed code works.\n\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

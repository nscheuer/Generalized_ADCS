"""The SSC26 Colab notebook must actually run.

``papers/SSC26_poster/SSC26_quickstart.ipynb`` is what the poster's "Runs in
your browser" QR code leads to. Nothing executed it, so the ``State`` migration
broke it in two places -- ``simulate(x=...)`` stopped accepting a flat
``[w, q, h]`` vector, and ``run.state_hist`` became a list of ``State`` rather
than a stackable array -- and it stayed broken, in public, behind a printed QR
code that cannot be recalled.

This runs the notebook's real cells over a **shortened horizon**. The point is
the API contract, not the physics: the documented 3000 s result is verified by
running the notebook itself, whereas a 120 s run exercises every call the
notebook makes -- construction, ``simulate``, ``state_hist`` access, the
pointing-error reduction and the plotting path -- in a couple of seconds, which
is cheap enough to run on every PR.

Companion to ``test_poster_snippets.py``, which does the same job for the code
printed on the poster.
"""

import json
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # no display in CI; plt.show() becomes a no-op

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "papers" / "SSC26_poster" / "SSC26_quickstart.ipynb"

# Long enough to converge is the notebook's job; this only has to reach every
# API call. Keep in sync with the notebook if that literal ever changes.
FULL_HORIZON = "tf=3000.0"
TEST_HORIZON = "tf=120.0"


def _notebook_source():
    """The notebook's code cells, minus the pip install, on a short horizon."""
    nb = json.loads(NOTEBOOK.read_text())
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "pip install" not in "".join(c["source"])
    ]
    assert cells, "no executable code cells found in the notebook"
    src = "\n".join(cells)
    assert FULL_HORIZON in src, (
        f"{FULL_HORIZON!r} no longer appears in the notebook; update "
        "TEST_HORIZON in this test or it will run the full simulation"
    )
    return src.replace(FULL_HORIZON, TEST_HORIZON)


def test_notebook_exists():
    assert NOTEBOOK.is_file(), (
        f"{NOTEBOOK.relative_to(REPO_ROOT)} is missing -- the poster's "
        "'Runs in your browser' QR code points at it."
    )


def test_notebook_cells_execute():
    """Every cell runs, and the pointing error it computes is real."""
    ns: dict = {"__name__": "__notebook__"}
    try:
        exec(compile(_notebook_source(), str(NOTEBOOK), "exec"), ns)
    except Exception as exc:  # noqa: BLE001 -- want the original traceback text
        pytest.fail(
            "The SSC26 quickstart notebook no longer runs. It is behind a "
            "printed QR code on poster SSC26-P2-54 and cannot be recalled.\n"
            f"{type(exc).__name__}: {exc}"
        )

    err = np.asarray(ns["err"], float)
    assert err.size > 0, "notebook produced no pointing-error history"
    assert np.all(np.isfinite(err)), "notebook pointing error contains NaN/inf"
    # Boresight-to-target angle: must be a physical angle in degrees.
    assert err.min() >= 0.0 and err.max() <= 180.0, (
        f"pointing error outside [0, 180] deg: [{err.min()}, {err.max()}]")
    assert 89.0 < err[0] < 91.0, (
        f"notebook should start ~90 deg off target, got {err[0]:.2f}")

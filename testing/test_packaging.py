"""The declared package list must match what is on disk.

pyproject.toml lists packages explicitly rather than using `packages.find`,
because ADCS.pipeline's source lives at papers/Generalized_ACS/pipeline/ and
find cannot discover it there. The cost of an explicit list is that adding a
subpackage and forgetting to list it silently ships a broken wheel -- the code
imports fine from a checkout and is simply absent for anyone who pip-installs.

These tests make that failure loud.
"""

import pathlib
import sys
import tomllib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_SRC = REPO / "papers" / "Generalized_ACS" / "pipeline"


def _declared():
    with open(REPO / "pyproject.toml", "rb") as fh:
        return set(tomllib.load(fh)["tool"]["setuptools"]["packages"])


def _on_disk():
    found = {
        str(p.parent.relative_to(REPO)).replace("/", ".")
        for p in (REPO / "ADCS").rglob("__init__.py")
    }
    # the mapped pipeline tree, named as it is imported rather than as it is stored
    for p in PIPELINE_SRC.rglob("__init__.py"):
        rel = p.parent.relative_to(PIPELINE_SRC)
        found.add("ADCS.pipeline" if str(rel) == "." else f"ADCS.pipeline.{str(rel).replace('/', '.')}")
    return found


def test_no_package_is_missing_from_pyproject():
    missing = sorted(_on_disk() - _declared())
    assert not missing, (
        "these packages exist on disk but are not listed in pyproject.toml, so "
        f"they will NOT be installed: {missing}")


def test_no_declared_package_is_absent_from_disk():
    stale = sorted(_declared() - _on_disk())
    assert not stale, (
        f"pyproject.toml lists packages that no longer exist: {stale}")


def test_pipeline_lives_with_its_paper():
    """The move this configuration exists to support."""
    assert PIPELINE_SRC.is_dir(), "pipeline source should be under papers/Generalized_ACS/"
    assert not (REPO / "ADCS" / "pipeline").exists(), (
        "ADCS/pipeline/ should not exist in the source tree; it is mapped there "
        "only at install time")


def test_package_dir_maps_the_pipeline():
    with open(REPO / "pyproject.toml", "rb") as fh:
        pkg_dir = tomllib.load(fh)["tool"]["setuptools"]["package-dir"]
    assert pkg_dir.get("ADCS.pipeline") == "papers/Generalized_ACS/pipeline", (
        "the package-dir mapping is what puts ADCS/pipeline into the wheel")


def test_pipeline_imports_as_adcs_pipeline():
    """Import path is unchanged by the move -- the poster depends on this."""
    from ADCS.pipeline import PipelineController  # noqa: F401
    from ADCS.pipeline.control_law import ControlLaw, LawInterface  # noqa: F401
    from ADCS.pipeline.data import AllocationConfig  # noqa: F401


def test_source_checkout_resolves_pipeline_via_path_extension():
    """A clone has no ADCS/pipeline/ directory, so __path__ must cover it."""
    import ADCS
    import ADCS.pipeline

    resolved = pathlib.Path(ADCS.pipeline.__file__).resolve()
    if (REPO / "ADCS" / "pipeline").exists():
        pytest.skip("running against an installed layout, not a checkout")
    assert PIPELINE_SRC in resolved.parents or resolved.parent == PIPELINE_SRC, (
        f"ADCS.pipeline resolved to {resolved}, expected it under {PIPELINE_SRC}")

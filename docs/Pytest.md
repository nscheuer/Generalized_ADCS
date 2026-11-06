# Pytest

## Test all files
This is the process that GitHub Actions performs during any push or merge request.
```bash
cd testing/
pytest -v
```
This process takes about 3 minutes.

## Test specific file
```bash
cd testing/
pytest ./test_controllers/test_controller_mtq_w_rw.py
```

## Skip local-only tests
The Godot viewer test is marked `local` because it requires a local Godot
binary and viewer source tree. To run the regular Python suite without local-only
tests:

```bash
pytest -m "not local" testing/
```

To run only the local Godot test:

```bash
pytest -m local testing/test_viz/test_viewer_godot.py
```

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

## Test based on flags
The following flags are available:
- `slow`
```bash
cd testing/
pytest -k "not slow"
```
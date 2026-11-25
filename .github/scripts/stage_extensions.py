"""Stage Python extension binaries while preserving their package paths."""

import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
count = 0
for path in source.rglob("*"):
    if "target" in path.parts or not path.is_file():
        continue
    if path.suffix not in {".so", ".pyd"}:
        continue
    target = destination / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    count += 1
if not count:
    raise SystemExit(f"No Python extensions found below {source}")
print(f"Staged {count} extensions from {source}")

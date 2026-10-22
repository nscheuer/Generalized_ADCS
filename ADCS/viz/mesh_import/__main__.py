"""``python -m ADCS.viz.mesh_import CONFIG --out DIR [--satellite module:factory] [--strict]``."""

import argparse
import importlib
import sys


def _load_satellite(spec: str):
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--satellite must look like package.module:factory")
    obj = getattr(importlib.import_module(module), attr)
    return obj() if callable(obj) else obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ADCS.viz.mesh_import",
                                 description="Import and check a spacecraft model for the 3-D viewer")
    ap.add_argument("config", help="spacecraft_viz.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--satellite", help="package.module:factory returning the Satellite to check against")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings")
    args = ap.parse_args(argv)

    from ADCS.viz.mesh_import import import_spacecraft

    sat = _load_satellite(args.satellite) if args.satellite else None
    res = import_spacecraft(args.config, args.out, satellite=sat)
    for line in res.info:
        print(f"  {line}")
    for line in res.warnings:
        print(f"warning: {line}")
    for line in res.errors:
        print(f"error: {line}", file=sys.stderr)
    if res.sidecar:
        print(f"wrote {res.sidecar} and {len(res.lods)} levels of detail")
    return res.exit_code(args.strict)


if __name__ == "__main__":
    sys.exit(main())

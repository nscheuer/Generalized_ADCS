"""``python -m ADCS.helpers.animations.godot.assets {list,fetch,build,verify,licenses,path}``."""

import argparse
import json
import sys

from ADCS.helpers.animations.godot.assets.fetch import cache_dir, fetch, load_sources, raw_path
from ADCS.helpers.animations.godot.assets.products import (
    PRODUCTS,
    build,
    current_products,
    verify_processed,
    write_attribution,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ADCS.helpers.animations.godot.assets", description="Visualization asset pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="sources and products, and what is cached")
    p = sub.add_parser("fetch", help="download and verify raw sources")
    p.add_argument("ids", nargs="*")
    p.add_argument("--accept-license", action="append", default=[], metavar="ID")
    p = sub.add_parser("fetch-planets", help="request optional planet textures")
    p.add_argument("names", nargs="*")
    p.add_argument("--lod", choices=("2k", "8k"), default="2k", help="texture resolution tier")
    p = sub.add_parser("build", help="build products (fetches what they need)")
    p.add_argument("products", nargs="*", help=f"any of {list(PRODUCTS)} (default: all)")
    p.add_argument("--set", action="append", default=[], metavar="PRODUCT.PARAM=VALUE",
                   help="override a parameter, e.g. earth_day.face_size=2048")
    sub.add_parser("verify", help="re-check raw and processed checksums")
    sub.add_parser("licenses", help="write ATTRIBUTION.txt for the current products")
    sub.add_parser("path", help="print the cache directory")
    args = ap.parse_args(argv)

    sources = load_sources()
    if args.cmd == "path":
        print(cache_dir())
    elif args.cmd == "list":
        print(f"cache: {cache_dir()}\n\nsources:")
        for sid, s in sources.items():
            have = "cached" if raw_path(sid, s).exists() else "-"
            print(f"  {sid:28s} {s.get('bytes', 0) / 1e6:7.1f} MB  {have:6s}  {s['attribution']}")
        from .planets import FILES
        print("\nSolar System Scope textures (fetch-planets --lod 2k|8k):")
        for body, tiers in FILES.items():
            status = []
            for lod, filename in tiers.items():
                texture_dir = cache_dir() / "planet_textures"
                cached_path = texture_dir / filename
                if filename.lower().endswith((".tif", ".tiff")):
                    cached_path = cached_path.with_suffix(".png")
                have = "cached" if cached_path.is_file() else "-"
                status.append(f"{lod}={filename} [{have}]")
            print(f"  {body:14s} " + ", ".join(status))
        current = current_products()
        print("\nproducts:")
        for name, prod in PRODUCTS.items():
            print(f"  {name:18s} {json.dumps(prod.params):40s} {'built' if name in current else '-'}  {prod.description}")
    elif args.cmd == "fetch":
        fetch(args.ids or None, sources=sources, accept=args.accept_license)
    elif args.cmd == "fetch-planets":
        from .planets import fetch_planets
        for name, path in fetch_planets(args.names or None, lod=args.lod).items():
            print(f"{name}: {path}")
    elif args.cmd == "build":
        unknown = [n for n in args.products if n not in PRODUCTS]
        if unknown:
            ap.error(f"unknown products {unknown}; choose from {list(PRODUCTS)}")
        overrides = {}
        for item in args.set:
            key, _, value = item.partition("=")
            prod, _, param = key.partition(".")
            if prod not in PRODUCTS or not param:
                ap.error(f"bad --set {item!r}")
            overrides.setdefault(prod, {})[param] = json.loads(value)
        build(args.products or None, overrides=overrides, sources=sources)
        print(f"attribution: {write_attribution(sources)}")
    elif args.cmd == "verify":
        fetch([sid for sid, s in sources.items() if raw_path(sid, s).exists()], sources=sources)
        bad = {k: v for k, v in verify_processed().items() if v}
        for k, v in bad.items():
            print(f"{k}: {'; '.join(v)}")
        print("raw and processed assets verified" if not bad else "problems found")
        return 1 if bad else 0
    elif args.cmd == "licenses":
        print(write_attribution(sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())

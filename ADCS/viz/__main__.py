"""Command line: ``python -m ADCS.viz {convert,info} ...``."""

import argparse
import sys
from pathlib import Path


def _convert(args: argparse.Namespace) -> int:
    from ADCS.helpers.simresults import SimulationResults
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.viz.from_results import results_to_vizlog

    ephem = Ephemeris(name=args.ephem)
    sim = SimulationResults.load(args.input, ephem=ephem)
    out = results_to_vizlog(sim, args.output, ephem=ephem, torques=args.torques, overwrite=args.overwrite)
    print(f"wrote {out}")
    return 0


def _info(args: argparse.Namespace) -> int:
    from ADCS.viz.reader import VizLog

    log = VizLog.open(args.path)
    m = log.manifest
    print(f"{log.path}: {m['format']} v{m['schema_version']}, epoch {m['epoch']['iso_utc']} (TT JD {log.epoch_jd_tt:.6f})")
    eph = m.get("ephemeris") or {}
    print(f"ephemeris: {eph.get('name')}  eop: {eph.get('eop')}")
    for rid in log.run_ids:
        n = log.n_rows(rid)
        t = log.read(rid, "t_s")
        span = f"{t[0, 0]:.3f} .. {t[-1, 0]:.3f} s" if n else "empty"
        print(f"run {rid}: {n} rows, {span}{'' if log.is_complete(rid) else ' (incomplete)'}")
    width = max(len(c) for c in log.channels)
    for c in log.channels.values():
        print(f"  {c.name:<{width}}  {c.width:>2}  {c.unit:<8} {c.frame}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ADCS.viz", description="ADCS visualization tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert a saved .sim results file to an adcs-vizlog directory")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--ephem", default="de421", choices=["de421", "de440s", "de442s"],
                   help="ephemeris for recomputed quantities (default: de421, as the simulation)")
    p.add_argument("--torques", default="nominal", choices=["nominal", "none"])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_convert)

    p = sub.add_parser("info", help="summarize an adcs-vizlog directory")
    p.add_argument("path", type=Path)
    p.set_defaults(func=_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

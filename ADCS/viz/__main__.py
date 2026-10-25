"""Command line: ``python -m ADCS.viz {convert,info,view,render,listen,install-viewer} ...``."""

import argparse
import sys
from pathlib import Path


def _convert(args: argparse.Namespace) -> int:
    from ADCS.helpers.simresults import SimulationResults
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.viz.from_results import results_to_vizlog

    ephem = Ephemeris(name=args.ephem)
    sim = SimulationResults.load(args.input, ephem=ephem)
    out = results_to_vizlog(sim, args.output, ephem=ephem, torques=args.torques, overwrite=args.overwrite,
                            model=args.model)
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


def _view(args: argparse.Namespace) -> int:
    from ADCS.viz.binary_manager import ViewerNotFoundError
    from ADCS.viz.launcher import view

    try:
        return view(args.path, binary=args.viewer, host=args.host, port=args.port, launch=not args.no_launch,
                    project=args.viewer_project, assets=args.assets)
    except ViewerNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _listen(args: argparse.Namespace) -> int:
    from ADCS.viz.client import VizClient
    from ADCS.viz.protocol import MsgType

    host, _, port = args.address.rpartition(":")
    with VizClient(host or "127.0.0.1", int(port), args.token) as client:
        hello = client.connect()
        m = hello["manifest"]
        print(f"connected: mode {hello['mode']}, epoch {m['epoch']['iso_utc']}, "
              f"{len(client.names)} channels, scene {hello['scene'].get('spacecraft', {}).get('mass_kg')} kg")
        rows = 0
        for kind, data in client.messages():
            if kind == MsgType.BLOCK:
                rid, start, cols = data
                rows += len(cols["t_s"])
                print(f"block run {rid}: rows {start}..{start + len(cols['t_s']) - 1}")
            elif kind == MsgType.ROW:
                rows += 1
                rid, cols = data
                if args.verbose:
                    print(f"row run {rid}: t={cols['t_s'][0]:.3f} s  r={cols['r_sc'].round(3).tolist()} km")
            elif kind == MsgType.END:
                print(f"end of run {data}")
        print(f"session ended after {rows} rows")
    return 0


def _render(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from ADCS.viz.binary_manager import ViewerNotFoundError
    from ADCS.viz.render import RenderJob, load_job, parse_key, render_video

    job = load_job(args.job) if args.job else RenderJob()
    w, h = (int(x) for x in args.res.split("x")) if args.res else (job.width, job.height)
    over = {k: v for k, v in {"duration_s": args.duration, "fps": args.fps, "t_start": args.start, "t_end": args.end,
                              "supersample": args.supersample, "motion_blur": args.motion_blur,
                              "quality": args.quality}.items() if v is not None}
    job = replace(job, width=w, height=h, ui=job.ui or args.ui, **over)
    if args.key:
        job = replace(job, keys=[parse_key(k) for k in args.key])
    try:
        res = render_video(args.path, job, args.out, binary=args.viewer, project=args.viewer_project,
                           assets=args.assets, codec=args.codec, crf=args.crf, keep_frames=args.keep_frames,
                           run_id=args.run)
    except (ViewerNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(res.video or res.frames_dir)
    return 0


def _install_viewer(args: argparse.Namespace) -> int:
    from ADCS.viz.binary_manager import ViewerNotFoundError, install_viewer

    try:
        print(install_viewer())
    except ViewerNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
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
    p.add_argument("--model", type=Path, help="spacecraft model sidecar (.viz.json) from ADCS.viz.mesh_import")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_convert)

    p = sub.add_parser("info", help="summarize an adcs-vizlog directory")
    p.add_argument("path", type=Path)
    p.set_defaults(func=_info)

    p = sub.add_parser("view", help="serve a vizlog and open the 3-D viewer on it")
    p.add_argument("path", type=Path)
    p.add_argument("--viewer", type=Path, help="viewer executable (default: ADCS_VIEWER_BIN or installed release)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-launch", action="store_true",
                   help="only serve; print the arguments for a viewer started by hand")
    p.add_argument("--viewer-project", type=Path,
                   help="Godot project when --viewer is a plain Godot binary (default: the repo's viewer/)")
    p.add_argument("--assets", type=Path, help="asset cache (default: python -m ADCS.viz.assets path)")
    p.set_defaults(func=_view)

    p = sub.add_parser("render", help="render a vizlog to a video (offline, frame-exact)",
                       description="Examples:\n  python -m ADCS.viz render run.vizlog run.mp4 --duration 10 --res 1920x1080 "
                       "--supersample 2 --motion-blur 4\n  python -m ADCS.viz render run.vizlog fly.mp4 "
                       "--key at=0,dist=1.5 --key at=8,dist=2e6,yaw=-60,pitch=30\n  python -m ADCS.viz render run.vizlog "
                       "out.mp4 --job job.json", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("out", type=Path, help="output video (.mp4 / .mov), or a directory with --codec none")
    p.add_argument("--job", type=Path, help="RenderJob as JSON (fields of ADCS.viz.render.RenderJob)")
    p.add_argument("--duration", type=float, help="video length [s] (default 10)")
    p.add_argument("--fps", type=float)
    p.add_argument("--res", help="WxH, e.g. 1920x1080")
    p.add_argument("--start", type=float, help="simulation time of the first frame [s]")
    p.add_argument("--end", type=float, help="simulation time of the last frame [s]")
    p.add_argument("--supersample", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--motion-blur", type=int, help="sub-frames per frame (1 = none)")
    p.add_argument("--quality", choices=["high", "balanced", "fast"])
    p.add_argument("--ui", action="store_true", help="draw the timeline and HUD into the video")
    p.add_argument("--key", action="append", help="camera key: at=T,mode=orbit|earth,frame=lvlh|inertial|body,"
                   "dist=M,yaw=DEG,pitch=DEG,fov=DEG,ease=smooth|linear (repeat)")
    p.add_argument("--codec", default="h264", choices=["h264", "h265", "prores", "none"])
    p.add_argument("--crf", type=int, default=16, help="h264/h265 quality (lower is better)")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--run", help="run id (default: the first)")
    p.add_argument("--viewer", type=Path)
    p.add_argument("--viewer-project", type=Path)
    p.add_argument("--assets", type=Path)
    p.set_defaults(func=_render)

    p = sub.add_parser("listen", help="reference client: connect to a server and print what it sends")
    p.add_argument("address", help="host:port")
    p.add_argument("--token", required=True)
    p.add_argument("-v", "--verbose", action="store_true", help="print every live row")
    p.set_defaults(func=_listen)

    p = sub.add_parser("install-viewer", help="download and verify the pinned viewer release")
    p.set_defaults(func=_install_viewer)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

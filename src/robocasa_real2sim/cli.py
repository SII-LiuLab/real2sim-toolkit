import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .adapters import build_native
from .appearance import rank_textures
from .capture import inspect_captures
from .catalog import native_catalog, scan_local_assets, select_assets
from .io import read_request, write_json
from .reconstruction import find_colmap, reconstruction_plan, run_reconstruction
from .validation import validate_asset
from .preview import render_preview
from .isaac_prepare import prepare_mjcf
from .usd_backend import build_mesh_usd, inspect_usd, prepare_usd
from .deformable import build_deformable
from .deformable_sim import fit_deformable, validate_deformable


def doctor(colmap_path=None):
    executable = find_colmap(colmap_path)
    colmap = None
    if executable:
        result = subprocess.run([executable, "-h"], capture_output=True, text=True, timeout=30)
        colmap = {"executable": executable, "working": result.returncode == 0,
                  "version": (result.stdout + result.stderr).splitlines()[:2]}
    return {"python": sys.executable,
            "modules": {name: importlib.util.find_spec(name) is not None for name in
                        ("robocasa", "mujoco", "PIL", "trimesh", "tetgen", "pxr", "isaacsim", "robosimstudio", "newton", "ovrtx", "ovstage")},
            "colmap": colmap,
            "reference_tools": {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}}


def candidates_for(args):
    candidates = native_catalog()
    if args.assets:
        candidates += scan_local_assets(args.assets)
    return candidates


def _pipeline(args, report):
    request = read_request(args.request)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    captures = inspect_captures(request)
    if request.get("behavior") == "deformable" or request.get("deformable"):
        write_json(output / "capture.json", captures)
        manifest = build_deformable(request, output / "asset")
        report.update(object_id=request["id"], route="deformable", status=manifest["status"],
                      capture="capture.json", asset="asset/manifest.json", dynamics_validated=False)
        write_json(output / "pipeline.json", report)
        return report
    selection = select_assets(request, candidates_for(args))
    write_json(output / "capture.json", captures)
    write_json(output / "selection.json", selection)
    report.update(object_id=request["id"], route=selection["route"],
                  capture="capture.json", selection="selection.json")
    if selection["candidates"]:
        if args.candidate:
            matches = [c for c in selection["candidates"] if c["id"] == args.candidate]
            if not matches:
                raise ValueError("Chosen candidate failed the structural filters")
            candidate = matches[0]
        else:
            candidate = selection["candidates"][0]
        if candidate["source"] != "robocasa":
            report.update(status="external_asset_review_required", candidate=candidate)
        else:
            appearance = rank_textures(request)
            write_json(output / "appearance.json", appearance)
            report["appearance"] = "appearance.json"
            if (request.get("appearance", {}).get("apply_best_texture") and
                    appearance["candidates"] and not request.get("geometry", {}).get("texture")):
                request.setdefault("geometry", {})["texture"] = appearance["candidates"][0]["texture"]
                request["appearance"]["selection_source"] = "local surface texture heuristic"
                report["selected_texture"] = request["geometry"]["texture"]
            manifest = build_native(request, candidate, output / "asset")
            validation = validate_asset(output / "asset/model.xml")
            report.update(status="preview_asset_built", asset="asset/model.xml",
                          real_world_calibrated=manifest["real_world_calibrated"], validation=validation)
    else:
        if not args.state:
            raise ValueError("Reconstruction requires --state to keep moving parts separate")
        plan = reconstruction_plan(captures, args.state, args.part)
        if args.run_reconstruction:
            plan = run_reconstruction(plan, output / "reconstruction", args.colmap, args.dense, jobs_root=args.jobs_root, resume=args.resume)
        else:
            write_json(output / "reconstruction/reconstruction.json", plan)
        report.update(status=plan["status"], reconstruction="reconstruction/reconstruction.json")
    write_json(output / "pipeline.json", report)
    return report


def pipeline(args):
    report = {"status": "running"}
    report_path = args.output.resolve() / "pipeline.json"
    write_json(report_path, report)
    try:
        return _pipeline(args, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise
    except KeyboardInterrupt:
        report.update(status="interrupted")
        write_json(report_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Asset-first real2sim for household and industrial assets")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Check dependencies in the current Python environment")
    p.add_argument("--colmap")
    p = commands.add_parser("runtime-status", help="Inspect bundled RoboSimStudio provenance and pinned runtime dependencies without loading GPU libraries")
    p.add_argument("--renderer", choices=["tiled", "rtx"], default="tiled")
    p = commands.add_parser("agent", help="Open or reuse a reconstruction session's interactive agent workbench")
    p.add_argument("--session", type=Path)
    p.add_argument("--describe", action="store_true", help="Print the coding-agent tool contract without starting a simulator")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--renderer", choices=["tiled", "rtx"], default="tiled")
    p.add_argument("--panels", choices=["all", "drive"], default="drive")
    p.add_argument("--resume", action="store_true", help="Start/resume physics; otherwise preserve an existing page's pause state")
    p.add_argument("--compact", action="store_true", help="Return bounded previews and report references instead of full metadata")
    p = commands.add_parser("session-init", help="Create an editable reconstruction session; preserve it on rerun")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--request", type=Path)
    p.add_argument("--reference", type=Path, action="append", default=[])
    for name in ("session-status", "session-log"):
        p = commands.add_parser(name, help="Inspect the latest attempts or immutable operation history")
        p.add_argument("--session", type=Path, required=True)
    p = commands.add_parser("session-brief", help="Restore current goal, usable memories, failures and artifact locations in one bounded response")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--probe", action="store_true", help="Also probe the existing native page; never start or reset it")
    p.add_argument("--limit", type=int, default=5)
    p = commands.add_parser("session-focus", help="Record the current request and invalidate next actions from an older scope")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--goal", required=True)
    p.add_argument("--mode", choices=["asset", "scene", "interaction", "tooling"], required=True)
    p.add_argument("--constraint", action="append", default=[])
    p.add_argument("--validation", choices=["unspecified", "none", "runtime", "visual", "interaction"], default="unspecified")
    p = commands.add_parser("session-checkpoint", help="Save a public milestone and next action for the current focus revision")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--stage", choices=["inspect", "author", "preview", "validate", "deliver", "blocked"], required=True)
    p.add_argument("--note", required=True)
    p.add_argument("--focus-revision", type=int, required=True)
    p.add_argument("--next-action")
    p.add_argument("--evidence", type=Path, action="append", default=[])
    p = commands.add_parser("memory-put", help="Store an explicit decision, constraint or evidenced fact without a chat transcript")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--kind", choices=["goal", "constraint", "decision", "fact", "failed_approach"], required=True)
    p.add_argument("--content", required=True)
    p.add_argument("--source", choices=["user", "observed", "inferred"], required=True)
    p.add_argument("--source-note")
    p.add_argument("--depends-on-scene", action="store_true")
    p.add_argument("--scope", choices=["focus", "session"], help="Default: focus for goals/constraints, session for other kinds")
    p.add_argument("--evidence", action="append", default=[], help="Session-relative evidence file or explicit URL")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--supersedes", help="Exact current record ID when intentionally replacing a key")
    p = commands.add_parser("memory-recall", help="Recall relevant active memories; changed evidence and old focus are excluded by default")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--query")
    p.add_argument("--kind", choices=["goal", "constraint", "decision", "fact", "failed_approach"])
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--include-stale", action="store_true")
    p = commands.add_parser("memory-forget", help="Retire a memory key while retaining its explicit history")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--key", required=True)
    p = commands.add_parser("session-serve", help="Open RoboSimStudio's live workbench over the editable session")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--renderer", choices=["tiled", "rtx"], default="tiled")
    p.add_argument("--panels", choices=["all", "drive"], default="all", help="Use drive for a compact asset manipulation page")
    p.add_argument("--resume", action="store_true", help="Start physics after opening the page; default starts paused")
    p = commands.add_parser("session-ctl", help="Control this session's live page with RoboSimStudio's native commands")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER, help="Native command, e.g. status, snapshot, play, reset, log")
    p = commands.add_parser("ref-frames", help="Copy reference inputs and extract video review frames")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("source", type=Path)
    p.add_argument("--times", type=float, nargs="+", help="Explicit increasing timestamps in seconds")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--compact", action="store_true")
    for name in ("scene-build", "scene-run", "scene-snapshot", "scene-render"):
        p = commands.add_parser(name, help="Execute the session's authored RoboSimStudio scene in an isolated process")
        p.add_argument("--session", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--renderer", choices=["tiled", "rtx"], default="tiled")
        p.add_argument("--timeout", type=float, default=300, help="Worker time limit in seconds")
        p.add_argument("--compact", action="store_true", help="Keep the full report on disk and return a bounded preview")
        if name in {"scene-run", "scene-snapshot"}:
            p.add_argument("--steps", type=int, default=120 if name == "scene-run" else 0,
                           help="Control-step budget; each command starts from a fresh scene")
            p.add_argument("--routine", nargs="?", const="routine", help="Run the default or a named authored generator")
            p.add_argument("--camera", help="Named scene camera")
        if name == "scene-run":
            p.add_argument("--capture-every", type=int, default=0, help="Render every N steps; --record defaults to every step")
            p.add_argument("--record", action="store_true", help="Write rollout.mp4 alongside captured frames")
            p.add_argument("--video-fps", type=float, help="Override playback rate; default follows simulation capture timing")
        if name == "scene-render":
            p.add_argument("--trajectory", type=Path, required=True, help="Captured trajectory.npz, including rigid states when the scene has robots")
            p.add_argument("--model-geometry", type=Path, required=True, help="Matching recorded model_geometry.npz")
            p.add_argument("--camera", help="Named scene camera")
            p.add_argument("--every", type=int, default=2, help="Render every N recorded states and the final state, without advancing physics")
            p.add_argument("--record", action="store_true", help="Write replay.mp4 alongside the rendered PNGs")
            p.add_argument("--video-fps", type=float, help="Override playback rate; default follows recorded timestamps")
    p = commands.add_parser("scene-compare", help="Review images and optionally measure explicit aligned silhouette masks")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--rendered", type=Path, required=True)
    p.add_argument("--reference-mask", type=Path)
    p.add_argument("--rendered-mask", type=Path)
    p.add_argument("--min-iou", type=float, help="Optional silhouette-only pass threshold")
    p.add_argument("--compact", action="store_true")
    p = commands.add_parser("catalog", help="List native and explicitly labelled local assets")
    p.add_argument("--assets", type=Path)
    for name in ("inspect", "select", "pipeline", "reconstruct"):
        p = commands.add_parser(name)
        p.add_argument("--request", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name in ("select", "pipeline"):
            p.add_argument("--assets", type=Path)
        if name in ("pipeline", "reconstruct"):
            p.add_argument("--state")
            p.add_argument("--part", default="assembly")
            p.add_argument("--colmap")
            p.add_argument("--jobs-root", type=Path)
            p.add_argument("--dense", action="store_true")
            p.add_argument("--resume", action="store_true")
        if name == "pipeline":
            p.add_argument("--candidate")
            p.add_argument("--run-reconstruction", action="store_true")
    p = commands.add_parser("validate")
    p.add_argument("model", type=Path)
    p = commands.add_parser("preview", help="Render exported fixture in closed/open states")
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("build-usd", help="Export metric, aligned household/industrial meshes")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("build-deformable", help="Prepare a cloth rest surface or tetrahedral soft body")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("validate-deformable", help="Run Newton VBD dynamics under authored supports/loads")
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=1.0, help="Seconds, a multiple of simulation timestep")
    p.add_argument("--min-deformation", type=float, default=1e-6, help="Minimum shape residual after rigid alignment, metres")
    p.add_argument("--max-edge-strain", type=float, default=1.0)
    p.add_argument("--observations", type=Path, help="Metric vertex tracks NPZ; sample times replace --duration")
    p.add_argument("--max-rmse", type=float, default=.01, help="Maximum observation RMSE in metres")
    p.add_argument("--device", default="cpu", help="Explicit Warp device, e.g. cpu or cuda:0; no silent fallback")
    p = commands.add_parser("fit-deformable", help="Fit Young's modulus to tracked deformation with known supports/loads")
    p.add_argument("bundle", type=Path)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--young-pa", type=float, nargs="+", required=True, help="2 to 100 positive candidate moduli in Pa")
    p.add_argument("--device", default="cpu", help="Explicit Warp device, e.g. cpu or cuda:0")
    p = commands.add_parser("isaac-prepare", help="Bundle expanded MJCF and freeze compiled body inertia")
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("usd-inspect", help="Inspect USD structure (no runtime physics test)")
    p.add_argument("model", type=Path)
    p = commands.add_parser("prepare-usd", help="Bundle local USD dependencies and optionally configure a belt")
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--belt-prim")
    p.add_argument("--speed", type=float, help="Signed belt surface speed in m/s")
    p.add_argument("--direction", type=float, nargs=3, default=[1, 0, 0], help="Direction in belt local coordinates")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.colmap)
        elif args.command == "runtime-status":
            from .runtime_env import runtime_report
            result = runtime_report(renderer=args.renderer)
        elif args.command == "agent":
            from .agent import agent_contract, open_agent
            if args.describe:
                result = agent_contract()
            elif args.session is None:
                raise ValueError("agent requires --session PATH, or use --describe for its tool contract")
            else:
                result = open_agent(args.session, port=args.port, device=args.device, renderer=args.renderer,
                                    panels=args.panels, resume=args.resume)
        elif args.command == "session-init":
            from .session import init_session
            result = init_session(args.output, args.request, args.reference)
        elif args.command == "session-status":
            from .session import session_status
            result = session_status(args.session)
        elif args.command == "session-brief":
            from .context import session_brief
            result = session_brief(args.session, probe=args.probe, limit=args.limit)
        elif args.command == "session-focus":
            from .context import set_focus
            result = set_focus(args.session, goal=args.goal, mode=args.mode,
                               constraints=args.constraint, validation=args.validation)
        elif args.command == "session-checkpoint":
            from .context import checkpoint
            result = checkpoint(args.session, stage=args.stage, note=args.note,
                                focus_revision=args.focus_revision, next_action=args.next_action, evidence=args.evidence)
        elif args.command == "memory-put":
            from .memory import put_memory
            source = {"type": args.source}
            if args.source_note:
                source["note"] = args.source_note
            if args.depends_on_scene:
                source["depends_on_scene"] = True
            scope = args.scope or ("focus" if args.kind in {"goal", "constraint"} else "session")
            evidence = [{"url": value} if value.startswith(("https://", "http://")) else value
                        for value in args.evidence]
            result = put_memory(args.session, key=args.key, kind=args.kind, content=args.content,
                                source=source, evidence=evidence, tags=args.tag,
                                supersedes=args.supersedes, scope=scope)
        elif args.command == "memory-recall":
            from .memory import recall_memory
            result = {"memories": recall_memory(args.session, query=args.query, kind=args.kind,
                                                limit=args.limit, include_stale=args.include_stale)}
        elif args.command == "memory-forget":
            from .memory import forget_memory
            result = forget_memory(args.session, args.key)
        elif args.command == "session-serve":
            from .live import serve_session
            serve_session(args.session, port=args.port, device=args.device, renderer=args.renderer,
                          panels=args.panels, resume=args.resume)
            return 0
        elif args.command == "session-ctl":
            from .live import control_session
            return control_session(args.session, args.arguments)
        elif args.command == "session-log":
            from .session import session_status
            session = session_status(args.session)
            path = args.session / session["log"]
            result = {"events": [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]}
        elif args.command == "ref-frames":
            from .harness import reference_frames
            result = reference_frames(args.session, args.source, args.times, args.fps, args.max_frames)
        elif args.command in {"scene-build", "scene-run", "scene-snapshot", "scene-render"}:
            from .harness import scene_operation
            result = scene_operation(args.session, args.command, device=args.device,
                                     steps=getattr(args, "steps", 0), routine=getattr(args, "routine", None),
                                     capture_every=getattr(args, "capture_every", 0),
                                     camera=getattr(args, "camera", None), renderer=args.renderer, timeout_s=args.timeout,
                                     record=getattr(args, "record", False), video_fps=getattr(args, "video_fps", None),
                                     trajectory=getattr(args, "trajectory", None), model_geometry=getattr(args, "model_geometry", None),
                                     every=getattr(args, "every", 2))
        elif args.command == "scene-compare":
            from .harness import compare_scene
            result = compare_scene(args.session, args.reference, args.rendered,
                                   reference_mask=args.reference_mask, rendered_mask=args.rendered_mask,
                                   min_iou=args.min_iou)
        elif args.command == "catalog":
            result = candidates_for(args)
        elif args.command == "validate":
            result = validate_asset(args.model)
        elif args.command == "preview":
            result = render_preview(args.model, args.output)
        elif args.command == "pipeline":
            result = pipeline(args)
        elif args.command == "build-usd":
            result = build_mesh_usd(read_request(args.request), args.output)
        elif args.command == "build-deformable":
            result = build_deformable(read_request(args.request), args.output)
        elif args.command == "validate-deformable":
            result = validate_deformable(args.bundle, args.output, args.duration, args.min_deformation,
                                         args.max_edge_strain, args.observations, args.max_rmse, args.device)
        elif args.command == "fit-deformable":
            result = fit_deformable(args.bundle, args.observations, args.output, args.young_pa, args.device)
        elif args.command == "isaac-prepare":
            result = prepare_mjcf(args.model, args.output)
        elif args.command == "usd-inspect":
            result = inspect_usd(args.model)
        elif args.command == "prepare-usd":
            if args.belt_prim and args.speed is None:
                raise ValueError("--speed is required with --belt-prim")
            result = prepare_usd(args.model, args.output, args.belt_prim, args.speed, args.direction)
        else:
            request = read_request(args.request)
            if args.command == "inspect":
                result = inspect_captures(request)
                write_json(args.output, result)
            elif args.command == "select":
                result = select_assets(request, candidates_for(args))
                if any(item["source"] == "robocasa" for item in result["candidates"]):
                    result["appearance"] = rank_textures(request)
                write_json(args.output, result)
            else:
                if not args.state:
                    raise ValueError("--state is required for static multiview reconstruction")
                plan = reconstruction_plan(inspect_captures(request), args.state, args.part)
                result = run_reconstruction(plan, args.output, args.colmap, args.dense, jobs_root=args.jobs_root, resume=args.resume)
        if getattr(args, "compact", False):
            from .context import compact_result
            result = compact_result(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2 if isinstance(result, dict) and result.get("status") in {"failed", "mismatch"} else 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        print(f"real2sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

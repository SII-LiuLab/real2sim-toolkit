import argparse
import importlib.util
import json
from pathlib import Path
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


def doctor(colmap_path=None):
    executable = find_colmap(colmap_path)
    colmap = None
    if executable:
        result = subprocess.run([executable, "-h"], capture_output=True, text=True, timeout=30)
        colmap = {"executable": executable, "working": result.returncode == 0,
                  "version": (result.stdout + result.stderr).splitlines()[:2]}
    return {"python": sys.executable,
            "modules": {name: importlib.util.find_spec(name) is not None for name in ("robocasa", "mujoco", "PIL", "trimesh", "pxr", "isaacsim")},
            "colmap": colmap}


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
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        print(f"real2sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

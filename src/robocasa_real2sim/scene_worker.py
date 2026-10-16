"""Isolated RoboSimStudio execution. Invoked by the durable harness tools."""

import argparse
from pathlib import Path
import traceback

from .io import read_json, write_json


def execute(settings):
    from .rss_backend import load_scene
    output = Path(settings["output"])
    operation = settings["operation"]
    if operation not in {"scene-build", "scene-run", "scene-snapshot", "scene-render"}:
        raise ValueError(f"Unknown scene operation: {operation}")
    with load_scene(settings["scene"], device=settings["device"],
                    renderer=settings["renderer"], force_newton=True) as runtime:
        build = runtime.build_report()
        result = {"status": "built", "build": build, "reconstruction_validated": False,
                  "review_required": True}
        if not build["finite"]["passed"]:
            return {**result, "status": "failed", "error": "Initial simulation state is non-finite"}
        if operation in {"scene-run", "scene-snapshot"}:
            steps = settings.get("steps", 0)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a nonnegative integer")
            if operation == "scene-run" and steps == 0:
                raise ValueError("scene-run requires steps > 0")
            if steps:
                result["run"] = runtime.run(steps, routine=settings.get("routine"),
                                            capture_every=settings.get("capture_every", 0),
                                            output_dir=output, camera=settings.get("camera"),
                                            video=settings.get("record", False),
                                            fps=settings.get("video_fps"))
                result["status"] = "executed"
                if result["run"].get("passed") is False or result["run"].get("status") == "failed":
                    result.update(status="failed", error=result["run"].get("error", "Scene execution failed"))
        if operation == "scene-snapshot":
            result["snapshot"] = runtime.snapshot(output / "snapshot.png", camera=settings.get("camera"))
            if result["status"] != "failed":
                result["status"] = "rendered"
        if operation == "scene-render":
            from .replay import render_trajectory
            result["replay"] = render_trajectory(
                runtime, settings["trajectory"], settings["model_geometry"], output,
                camera=settings.get("camera"), every=settings.get("every", 2),
                record=settings.get("record", False), fps=settings.get("video_fps"))
            if result["replay"]["passed"]:
                result["status"] = "rendered"
            else:
                result.update(status="failed", error=result["replay"].get("error", "Recorded-state rendering failed"))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.input.parent / "result.json"
    write_json(output, {"status": "running", "reconstruction_validated": False})
    try:
        result = execute(read_json(args.input))
        write_json(output, result)
        return 2 if result.get("status") == "failed" else 0
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        write_json(output, {"status": "failed", "error": str(error),
                            "exception_type": type(error).__name__, "reconstruction_validated": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Durable tools for an agent's reference -> scene -> run -> compare loop.

Each scene operation builds from the editable source in a fresh process. A
session records evidence and attempts; it does not claim automatic perception
or replace the coding agent's modelling decisions.
"""

from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .io import read_json, write_json
from .session import (begin_operation, fail_operation, finish_operation,
                      scene_fingerprint)


def _envelope(root, attempt, summary):
    return {"status": attempt["status"], "operation": attempt["operation"],
            "attempt_id": attempt["id"], "session": str(root),
            "report": attempt["directory"] + "/operation.json", "summary": summary}


def _perform(session, operation, parameters, action):
    root = Path(session).resolve()
    attempt = begin_operation(root, operation, parameters)
    output = root / attempt["directory"]
    summary = {}
    try:
        summary = action(root, output)
        after = scene_fingerprint(root)
        summary["source_changed_during_operation"] = after["sha256"] != attempt["fingerprint"]["sha256"]
        artifacts = {p.relative_to(output).as_posix(): p.relative_to(root).as_posix()
                     for p in sorted(output.rglob("*")) if p.is_file() and p.name != "operation.json"}
        if summary.get("status") == "failed" or summary.get("passed") is False:
            error = summary.get("error", "Operation reported a failed check")
            saved = fail_operation(root, attempt["id"], error, summary=summary, artifacts=artifacts)
        else:
            saved = finish_operation(root, attempt["id"], summary=summary, artifacts=artifacts)
        return _envelope(root, saved, summary)
    except KeyboardInterrupt:
        fail_operation(root, attempt["id"], "Interrupted", summary={"interrupted": True})
        raise
    except Exception as error:
        summary.update(status="failed", error=str(error))
        artifacts = {p.relative_to(output).as_posix(): p.relative_to(root).as_posix()
                     for p in sorted(output.rglob("*")) if p.is_file() and p.name != "operation.json"}
        saved = fail_operation(root, attempt["id"], str(error), summary=summary, artifacts=artifacts)
        return _envelope(root, saved, summary)


def reference_frames(session, source, times_s=None, fps=1.0, max_frames=12):
    from .reference import extract_reference
    from .session import register_reference
    # Preserve the input before extracting frames, even when it lives elsewhere.
    try:
        registered = register_reference(session, source)
    except (OSError, ValueError, RuntimeError) as error:
        # Even an invalid replacement input must supersede an earlier success.
        def invalid_source(root, output):
            raise error
        return _perform(session, "ref-frames", {"source": str(source)}, invalid_source)
    parameters = {"source": registered, "times_s": times_s, "fps": fps, "max_frames": max_frames}

    def action(root, output):
        source_path = root / registered["path"]
        return extract_reference(source_path, output, times_s=times_s, fps=fps, max_frames=max_frames)

    return _perform(session, "ref-frames", parameters, action)


def _scene_process(root, output, operation, parameters):
    timeout = float(parameters.get("timeout_s", 300))
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise ValueError("timeout_s must be in (0, 3600]")
    settings = {"scene": str(root / "scene.py"), "output": str(output),
                "operation": operation, **parameters}
    write_json(output / "input.json", settings)
    command = [sys.executable, "-m", "robocasa_real2sim.scene_worker",
               "--input", str(output / "input.json")]
    environment = os.environ.copy()
    environment["REAL2SIM_OUTPUT"] = str(output)
    with (output / "runtime.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise RuntimeError(f"Scene operation timed out after {timeout:g} s; see {output / 'runtime.log'}") from error
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Scene worker exited {process.returncode} without a result; see {output / 'runtime.log'}")
    result = read_json(result_path)
    result["worker_returncode"] = process.returncode
    if process.returncode and result.get("status") != "failed":
        result.update(status="failed", error=f"Scene worker exited {process.returncode}; inspect runtime.log")
    return result


def scene_operation(session, operation, *, device="cuda:0", steps=0, routine=None,
                    capture_every=0, camera=None, renderer="tiled", timeout_s=300,
                    record=False, video_fps=None, trajectory=None, model_geometry=None, every=2):
    if operation not in {"scene-build", "scene-run", "scene-snapshot", "scene-render"}:
        raise ValueError("Unknown scene operation")
    parameters = {"device": device, "steps": steps, "routine": routine,
                  "capture_every": capture_every, "camera": camera,
                  "renderer": renderer, "timeout_s": timeout_s,
                  "record": record, "video_fps": video_fps}
    if operation == "scene-render":
        parameters.update(trajectory=str(trajectory) if trajectory is not None else None,
                          model_geometry=str(model_geometry) if model_geometry is not None else None,
                          every=every)

    def action(root, output):
        settings = dict(parameters)
        hashes = {}
        if operation == "scene-render":
            if isinstance(every, bool) or not isinstance(every, int) or every < 1:
                raise ValueError("every must be a positive integer")
            # Preserve immutable replay inputs independently of the earlier run
            # directory or any external file which can later be replaced.
            for name in ("trajectory", "model_geometry"):
                if not parameters[name]:
                    raise ValueError(f"scene-render requires {name}")
                source = Path(parameters[name]).resolve(strict=True)
                destination = output / "inputs" / f"{name}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                hashes[name] = sha256(destination.read_bytes()).hexdigest()
                settings[name] = str(destination)
        result = _scene_process(root, output, operation, settings)
        if hashes:
            result["input_sha256"] = hashes
        return result

    return _perform(session, operation, parameters, action)


def compare_scene(session, reference, rendered, *, reference_mask=None, rendered_mask=None, min_iou=None):
    from .comparison import compare_images
    parameters = {"reference": str(reference), "rendered": str(rendered),
                  "reference_mask": str(reference_mask) if reference_mask else None,
                  "rendered_mask": str(rendered_mask) if rendered_mask else None,
                  "min_iou": min_iou}

    def action(root, output):
        copied = {}
        fingerprints = {}
        for name in ("reference", "rendered", "reference_mask", "rendered_mask"):
            value = parameters[name]
            if value:
                source = Path(value).resolve()
                destination = output / "inputs" / (name + source.suffix.lower())
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied[name] = destination
                fingerprints[name] = sha256(destination.read_bytes()).hexdigest()
        report = compare_images(copied["reference"], copied["rendered"], output,
                                reference_mask=copied.get("reference_mask"),
                                rendered_mask=copied.get("rendered_mask"), min_iou=min_iou)
        report["input_sha256"] = fingerprints
        if report.get("silhouette_passed") is False:
            report.update(status="failed", error="Silhouette IoU is below the requested threshold")
        return report

    return _perform(session, "scene-compare", parameters, action)

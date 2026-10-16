"""Bounded working context for the external coding agent.

This is explicit task state, not a transcript or private reasoning store. It
does not start a simulator, infer a user's permissions, or promote evidence to
a reconstruction verdict.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .io import read_json
from .session import _atomic_json, session_status


MODES = ("asset", "scene", "interaction", "tooling")
STAGES = ("inspect", "author", "preview", "validate", "deliver", "blocked")
VALIDATION_LEVELS = ("unspecified", "none", "runtime", "visual", "interaction")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, name, maximum=2048):
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} UTF-8 bytes")
    return value.strip()


@contextmanager
def _control(root):
    import fcntl
    directory = root / ".agent"
    if directory.is_symlink():
        raise ValueError("Agent control directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    for name in ("control.lock", "focus.json", "checkpoint.json", "control-events.jsonl"):
        if (directory / name).is_symlink():
            raise ValueError("Agent control files must not be symlinks")
    with (directory / "control.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield directory
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _event(directory, event, record):
    with (directory / "control-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), "event": event, **record}, ensure_ascii=False) + "\n")


def set_focus(session, *, goal, mode, constraints=None, validation="unspecified"):
    """Replace current scope explicitly; old checkpoints cannot drive a new goal."""
    root = Path(session).resolve()
    session_status(root)
    goal = _text(goal, "goal", 4096)
    if mode not in MODES or validation not in VALIDATION_LEVELS:
        raise ValueError("Unknown focus mode or requested validation level")
    if constraints is not None and not isinstance(constraints, (list, tuple)):
        raise ValueError("constraints must be a list of explicit text constraints")
    constraints = list(dict.fromkeys(_text(c, "constraint", 1024) for c in (constraints or [])))
    if len(constraints) > 16:
        raise ValueError("At most 16 active constraints are supported")
    content = {"goal": goal, "mode": mode, "constraints": constraints, "validation": validation}
    with _control(root) as directory:
        path = directory / "focus.json"
        previous = read_json(path) if path.exists() else {}
        if previous and all(previous.get(key) == value for key, value in content.items()):
            return {**previous, "changed": False}
        record = {"schema_version": 1, "revision": previous.get("revision", 0) + 1,
                  **content, "updated_at": _now()}
        _atomic_json(path, record)
        _event(directory, "focus_changed", record)
        # Retain the old checkpoint as evidence, but its revision mismatch
        # makes it ineligible as the current next action.
        return {**record, "changed": True, "next_action": None}


def _evidence(root, paths):
    records = []
    for value in paths or []:
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve(strict=True)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Checkpoint evidence must be a file inside the session") from error
        if not path.is_file():
            raise ValueError("Checkpoint evidence must be a file")
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"path": relative, "sha256": digest.hexdigest()})
    if len(records) > 12:
        raise ValueError("At most 12 checkpoint evidence files are supported")
    return records


def checkpoint(session, *, stage, note, focus_revision, next_action=None, evidence=None):
    """Save a public milestone/next action with compare-and-set scope checking."""
    root = Path(session).resolve()
    manifest = session_status(root)
    if manifest["current_fingerprint"] is None:
        raise ValueError("Cannot checkpoint unverifiable scene source: " + manifest.get("fingerprint_error", "unknown source error"))
    if stage not in STAGES:
        raise ValueError("Unknown checkpoint stage")
    if isinstance(focus_revision, bool) or not isinstance(focus_revision, int):
        raise ValueError("focus_revision must be an integer from session-brief")
    note = _text(note, "note", 4096)
    if next_action is not None:
        next_action = _text(next_action, "next_action", 1024)
    if stage == "deliver" and next_action:
        raise ValueError("A delivered milestone cannot also prescribe unfinished work")
    references = _evidence(root, evidence)
    with _control(root) as directory:
        focus_path = directory / "focus.json"
        focus = read_json(focus_path) if focus_path.exists() else None
        if focus is None or focus["revision"] != focus_revision:
            raise ValueError("Focus changed or is missing; restore session-brief before saving a checkpoint")
        if stage == "validate" and focus["validation"] == "none":
            raise ValueError("Current focus explicitly excludes validation; update focus if the request changed")
        record = {"schema_version": 1, "focus_revision": focus_revision,
                  "stage": stage, "note": note, "next_action": next_action,
                  "evidence": references, "source_fingerprint": manifest["current_fingerprint"],
                  "updated_at": _now(), "claim": "workflow milestone, not physical calibration"}
        _atomic_json(directory / "checkpoint.json", record)
        _event(directory, "checkpoint_saved", record)
        return record


def _clip(value, limit=320):
    value = str(value)
    return value if len(value) <= limit else value[:limit] + "…"


def compact_result(value):
    """Reduce tool response size while keeping verdicts and report locations.

    Disk reports and Python operation results remain complete. Every shortened
    collection is explicit, so a preview never looks like the full evidence.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"expected", "packages"} and isinstance(item, dict):
                result[key] = {"omitted_from_compact_response": True,
                               "items": len(item) if hasattr(item, "__len__") else None}
            else:
                result[key] = compact_result(item)
        return result
    if isinstance(value, list):
        if len(value) > 6:
            return {"count": len(value), "preview": [compact_result(v) for v in value[:2]],
                    "remaining_in_full_report": len(value) - 2}
        return [compact_result(item) for item in value]
    if isinstance(value, str):
        return _clip(value, 800)
    return value


def _operation(root, item):
    summary = item.get("summary", {})
    run = summary.get("run") or summary.get("replay") or {}
    output = {"operation": item["operation"], "attempt_id": item["id"],
              "status": item["status"], "stale": item.get("stale", True),
              "started_at": item.get("started_at"), "report": item["directory"] + "/operation.json"}
    error = item.get("error") or summary.get("error") or run.get("error")
    if error:
        output["error"] = _clip(error, 500)
    keys = ("status", "passed", "execution_passed", "budget_completed", "routine_status", "completed_steps",
            "physics_steps", "runtime_state_unchanged", "trajectory", "model_geometry", "video")
    output["result"] = {k: run[k] for k in keys if k in run}
    build = summary.get("build") or (summary if "counts" in summary else {})
    if build:
        output["build"] = {key: build[key] for key in ("counts", "device", "versions") if key in build}
        output["build"]["finite_passed"] = build.get("finite", {}).get("passed")
        output["build"]["solvers"] = [{key: island[key] for key in ("domains", "entities", "solver") if key in island}
                                        for island in build.get("islands", [])[:4]]
    if summary.get("snapshot"):
        output["result"]["snapshot"] = summary["snapshot"].get("path")
    if summary.get("metrics"):
        output["result"]["silhouette_iou"] = summary["metrics"].get("silhouette_iou")
        output["result"]["silhouette_passed"] = summary.get("silhouette_passed")
    artifacts = item.get("artifacts", {})
    priority = [(name, path) for name, path in artifacts.items()
                if Path(path).suffix in {".mp4", ".npz", ".log"} or name.endswith(("report.json", "snapshot.png"))]
    selected = (priority or list(artifacts.items()))[:5]
    output["artifacts"] = [{"name": name, "path": path, "exists": (root / path).is_file()}
                           for name, path in selected]
    output["artifact_count"] = len(artifacts)
    return output


def session_brief(session, *, probe=False, limit=5):
    """Restore goal, usable memories, failures and artifact locations in one call."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 12:
        raise ValueError("limit must be in [1,12]")
    root = Path(session).resolve()
    manifest = session_status(root)
    directory = root / ".agent"
    focus = read_json(directory / "focus.json") if (directory / "focus.json").exists() else None
    saved = read_json(directory / "checkpoint.json") if (directory / "checkpoint.json").exists() else None
    restored = None
    if saved:
        reasons = []
        if not focus or saved["focus_revision"] != focus["revision"]:
            reasons.append("focus_changed")
        if manifest["current_fingerprint"] is None or saved.get("source_fingerprint") is None:
            reasons.append("scene_source_unverifiable")
        elif saved.get("source_fingerprint") != manifest["current_fingerprint"]:
            reasons.append("scene_source_changed")
        for reference in saved.get("evidence", []):
            try:
                current = _evidence(root, [reference["path"]])[0]
                if current["sha256"] != reference["sha256"]:
                    reasons.append("evidence_changed:" + reference["path"])
            except (OSError, ValueError):
                reasons.append("evidence_missing:" + reference["path"])
        restored = {**saved, "stale": bool(reasons), "stale_reasons": reasons}
        if reasons:
            restored["next_action"] = None
    from .memory import recall_memory
    memories = recall_memory(root, limit=100,
                             priority_kinds=("constraint", "goal", "decision", "failed_approach", "fact"))
    # Keep user constraints and durable choices ahead of fresh diagnostics.
    priority = {"constraint": 0, "goal": 1, "decision": 2, "failed_approach": 3, "fact": 4}
    memories.sort(key=lambda item: priority.get(item["kind"], 5))
    memory_preview = [{key: item[key] for key in ("id", "key", "kind", "scope", "stale") if key in item}
                      | {"content": _clip(item["content"], 500),
                         "source": {key: _clip(item["source"][key], 200) for key in ("type", "note") if key in item["source"]}}
                      for item in memories[:12]]
    operations = sorted(manifest["operations"].values(), key=lambda item: item.get("started_at", ""), reverse=True)
    recent = [_operation(root, item) for item in operations[:limit]]
    failures = [_operation(root, item) for item in operations if item["status"] == "failed"][:3]
    live = {"status": "not_recorded", "verification": "recorded_only"}
    served = manifest["operations"].get("scene-serve")
    if served:
        live["scene_source_stale"] = bool(served.get("stale", True))
        live["recorded_at"] = served.get("started_at")
        live_path = root / served["directory"] / "live.json"
        if live_path.is_file():
            recorded = read_json(live_path)
            live.update({key: recorded[key] for key in ("status", "url", "device", "renderer", "panels") if key in recorded})
        else:
            live["status"] = "starting_or_unavailable"
    if probe:
        from .agent import live_agent_status
        try:
            current = live_agent_status(root)
            live = {"verification": "native_status_probe", "status": "stopped"}
            if current is not None:
                status = current["live"]
                live.update({"status": current["status"], "url": current["url"],
                             "scene_source_stale": current["scene_source_stale"],
                             **{key: status[key] for key in ("paused", "finite", "step", "time", "last_error") if key in status}})
        except (OSError, RuntimeError, ValueError) as error:
            live = {"verification": "native_status_probe", "status": "unverified", "error": _clip(error)}
    cache = None
    for item in operations:
        summary = item.get("summary", {})
        build = summary.get("build", summary)
        environment = build.get("runtime_environment") or summary.get("runtime_environment")
        if environment:
            cache = {key: environment[key] for key in ("status", "renderer", "lock_digest") if key in environment}
            cache.update(verification="recorded_only", report=item["directory"] + "/operation.json")
            break
    recommendation = (restored.get("next_action") if restored and not restored["stale"] else None)
    if focus is None:
        recommendation = "Set the current goal and mode with session-focus before planning work."
    elif restored and restored["stale"]:
        recommendation = "Restore the changed goal/source and inspect reusable evidence before choosing a new action."
    result = {"status": "context_ready" if manifest["current_fingerprint"] is not None else "failed",
            "session": str(root), "scene": str(root / "scene.py"),
            "source_fingerprint": manifest["current_fingerprint"], "focus": focus,
            "checkpoint": restored, "next_action": recommendation, "memories": memory_preview,
            "recent_operations": recent, "latest_failures": failures, "live": live,
            "runtime": cache or {"status": "not_recorded", "next_check": "runtime-status only when runtime capability is needed"},
            "references": [{k: reference[k] for k in ("path", "kind") if k in reference}
                           for reference in manifest.get("references", [])[:5]],
            "omitted": {"operations": max(0, len(operations) - limit),
                        "memories": max(0, len(memories) - len(memory_preview)),
                        "references": max(0, len(manifest.get("references", [])) - 5)},
            "claims": "Context restoration only; recorded readiness and workflow stage do not establish reconstruction quality."}
    result["memory_more_possible"] = len(memories) == 100
    if manifest.get("fingerprint_error"):
        result["error"] = _clip(manifest["fingerprint_error"], 500)
        result["next_action"] = "Inspect the scene source error before resuming dependent work."
    result["full_context_files"] = ["session.json", ".agent/focus.json", ".agent/checkpoint.json", ".agent/memory/state.json"]
    result["memory_read_policy"] = "Use memory-recall for applicability checks; the raw memory index does not compute staleness."
    result["brief_truncated"] = False
    # Bound model-visible context independently of artifact/trajectory size.
    # Constraints are never silently truncated: shortened focus explicitly
    # requires reading the preserved full focus record before acting.
    def too_large():
        return len(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")) > 16000
    for key, minimum in (("recent_operations", 1), ("memories", 2), ("latest_failures", 1), ("references", 0)):
        while too_large() and len(result[key]) > minimum:
            result[key].pop()
            if key == "recent_operations":
                result["omitted"]["operations"] += 1
            elif key in result["omitted"]:
                result["omitted"][key] += 1
            result["brief_truncated"] = True
    if too_large() and restored:
        result["checkpoint"] = {key: restored.get(key) for key in ("stage", "focus_revision", "stale", "stale_reasons")}
        result["checkpoint"]["stale_reasons"] = [_clip(reason, 160) for reason in restored.get("stale_reasons", [])[:3]]
        result["checkpoint"]["full_record"] = ".agent/checkpoint.json"
        result["brief_truncated"] = True
    if too_large() and focus:
        result["focus"] = {key: focus[key] for key in ("revision", "mode", "validation")}
        result["focus"].update(goal=_clip(focus["goal"], 500), full_record=".agent/focus.json",
                              requires_full_record=True, constraint_count=len(focus["constraints"]))
        result["brief_truncated"] = True
    if too_large():
        for key in ("recent_operations", "latest_failures", "memories", "references"):
            result[key] = []
        result["brief_truncated"] = True
        result["next_action"] = "Read the preserved focus/checkpoint records and the latest operation report before acting."
    if too_large():
        raise ValueError("The context brief exceeds its 16 KB bound; inspect the preserved focus/checkpoint records directly")
    return result

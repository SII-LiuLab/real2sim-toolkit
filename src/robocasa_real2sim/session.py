"""Persistent evidence for agent-authored reconstruction scenes.

The session owns input snapshots and operation attempts; it never executes scene
code. Outputs belong inside an attempt directory, leaving previous evidence
available while only the latest attempt is advertised as current.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", "node_modules",
}
_STATE_FILES = {"session.json", "events.jsonl"}
_SCENE_TEMPLATE = '''"""Author this scene from the supplied reference and reconstruction request.

RoboSimStudio scene contract:
    import robosimstudio as rss
    scene = rss.Scene(...)  # Geometry, materials, constraints, and cameras.

Optional: BUILD = {...} supplies scene.build keyword arguments.
Optional: routine(sim) is a generator driving the reference interaction;
          routines = {"name": routine} exposes named generator routines.

Use explicit metric dimensions and record measured/assumed material values.
Flexible dynamics must use Newton; do not silently switch the requested target.
Read session.json for copied reference paths. Keep helpers, assets, and local
dependencies beside this file so each attempt can fingerprint its inputs.
"""

raise NotImplementedError(
    "Author scene.py from the supplied reference before building: define a "
    "top-level RoboSimStudio scene and, when needed, routine(sim)."
)
'''


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(root):
    path = Path(root).resolve() / "session.json"
    if not path.is_file():
        raise ValueError(f"Not a reconstruction session: {path.parent}; initialize it first")
    return json.loads(path.read_text(encoding="utf-8"))


def _event(root, event, **fields):
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), "event": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _relative(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Session paths must remain inside {root}: {path}") from exc


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_hash(source):
    if source.is_file():
        return _hash(source)
    files = []
    for directory, dirs, names in os.walk(source, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_DIRS)
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError("Copy reference directory dependencies instead of using directory symlinks")
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"Reference contains a non-file input: {path}")
            files.append((path.relative_to(source).as_posix(), _hash(path)))
    return hashlib.sha256(json.dumps(sorted(files), separators=(",", ":")).encode()).hexdigest()


def scene_fingerprint(root):
    """Hash scene.py plus local helpers/assets/inputs, excluding generated state.

    Dependencies must be copied into the session. External symlinks are rejected
    rather than silently omitted from provenance; virtualenvs and caches are not
    source dependencies. Runtime environment versions belong in operation reports.
    """
    root = Path(root).resolve()
    if not (root / "scene.py").is_file():
        raise ValueError(f"Missing scene.py in {root}")
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        relative_dir = Path(directory).relative_to(root)
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_DIRS and not (
            relative_dir == Path(".") and name in {"attempts", ".agent"}
        ))
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Copy directory dependencies into the session instead of using symlinks")
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo", ".tmp")) or (relative_dir == Path(".") and name in _STATE_FILES):
                continue
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"Scene contains a non-file input: {path}")
            relative = _relative(root, path)
            files.append({"path": relative, "sha256": _hash(path), "size": path.stat().st_size})
    files.sort(key=lambda item: item["path"])
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def init_session(root, request_path=None, references=()):
    """Create or reopen a session without overwriting its authored scene or inputs."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "session.json").exists():
        manifest = {
            "schema_version": 1, "created_at": _now(), "updated_at": _now(),
            "scene": "scene.py", "request": None, "references": [],
            "operations": {}, "log": "events.jsonl",
        }
        _atomic_json(root / "session.json", manifest)
        _event(root, "session_initialized")
    if not (root / "scene.py").exists():
        (root / "scene.py").write_text(_SCENE_TEMPLATE, encoding="utf-8")
    manifest = _read(root)
    if request_path is not None:
        source = Path(request_path).resolve()
        request = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Reconstruction request must be a JSON object")
        checksum = _hash(source)
        previous = manifest["request"]
        if previous and previous["sha256"] != checksum:
            raise ValueError("Session already has a different request; use a new session")
        if previous is None:
            destination = root / "inputs" / "request.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and _hash(destination) != checksum:
                raise ValueError(f"Refusing to overwrite existing input: {destination}")
            if source != destination:
                shutil.copy2(source, destination)
            manifest["request"] = {"path": "inputs/request.json", "source": str(source), "sha256": checksum}
            if "simulator" in request:
                manifest["request"]["simulator"] = request["simulator"]
            manifest["updated_at"] = _now()
            _atomic_json(root / "session.json", manifest)
            _event(root, "request_registered", path="inputs/request.json")
    for reference in references:
        register_reference(root, reference)
    return session_status(root)


def register_reference(root, source):
    """Copy one reference file or directory, retaining directory-relative resources."""
    root = Path(root).resolve()
    manifest = _read(root)
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"Reference does not exist: {source}")
    if source.is_dir() and (source == root or source in root.parents):
        raise ValueError("A reference directory cannot contain the session")
    checksum = _reference_hash(source)
    for record in manifest["references"]:
        if record["source"] == str(source) and record.get("sha256") == checksum:
            return record
    destination = root / "references" / uuid.uuid4().hex[:12] / source.name
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*_IGNORED_DIRS))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    record = {
        "source": str(source), "path": destination.relative_to(root).as_posix(),
        "kind": "directory" if source.is_dir() else "file", "sha256": checksum,
    }
    manifest["references"].append(record)
    manifest["updated_at"] = _now()
    _atomic_json(root / "session.json", manifest)
    _event(root, "reference_registered", path=record["path"])
    return record


def begin_operation(root, operation, parameters=None):
    """Invalidate the previous latest result before an operation can fail."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", operation):
        raise ValueError("Operation must be a simple lowercase name")
    root = Path(root).resolve()
    manifest = _read(root)
    attempt_id = uuid.uuid4().hex
    attempt = {
        "id": attempt_id, "operation": operation, "status": "running",
        "started_at": _now(), "finished_at": None,
        "directory": f"attempts/{operation}/{attempt_id}",
        "parameters": parameters or {}, "fingerprint": None, "summary": {}, "artifacts": {},
    }
    _atomic_json(root / attempt["directory"] / "operation.json", attempt)
    manifest["operations"][operation] = attempt
    manifest["updated_at"] = _now()
    _atomic_json(root / "session.json", manifest)
    _event(root, "operation_started", operation=operation, attempt_id=attempt_id)
    try:
        attempt["fingerprint"] = scene_fingerprint(root)
        _atomic_json(root / attempt["directory"] / "operation.json", attempt)
        manifest["operations"][operation] = attempt
        _atomic_json(root / "session.json", manifest)
    except Exception as exc:
        fail_operation(root, attempt_id, str(exc))
        raise
    return attempt


def _complete(root, attempt_id, status, summary, artifacts, error=None):
    root = Path(root).resolve()
    manifest = _read(root)
    candidates = list((root / "attempts").glob(f"*/{attempt_id}/operation.json")) if re.fullmatch(r"[a-f0-9]{32}", attempt_id) else []
    if len(candidates) != 1:
        raise ValueError(f"Unknown operation attempt: {attempt_id}")
    attempt = json.loads(candidates[0].read_text(encoding="utf-8"))
    if attempt["status"] != "running":
        raise ValueError(f"Operation attempt is already {attempt['status']}")
    portable_artifacts = {}
    for key, value in (artifacts or {}).items():
        relative = _relative(root, value)
        if not (root / relative).exists():
            raise ValueError(f"Artifact does not exist: {relative}")
        portable_artifacts[key] = relative
    attempt.update(status=status, finished_at=_now(), summary=summary or {}, artifacts=portable_artifacts)
    if error is not None:
        attempt["error"] = str(error)
    _atomic_json(candidates[0], attempt)
    # A slower earlier attempt may finish after its replacement. Keep it in the
    # history, but never let it supersede the operation the user most recently ran.
    if manifest["operations"][attempt["operation"]]["id"] == attempt_id:
        manifest["operations"][attempt["operation"]] = attempt
        manifest["updated_at"] = _now()
        _atomic_json(root / "session.json", manifest)
    _event(root, f"operation_{status}", operation=attempt["operation"], attempt_id=attempt_id)
    return attempt


def finish_operation(root, attempt_id, summary=None, artifacts=None):
    return _complete(root, attempt_id, "succeeded", summary, artifacts)


def fail_operation(root, attempt_id, error, summary=None, artifacts=None):
    return _complete(root, attempt_id, "failed", summary, artifacts, error)


def session_status(root):
    """Return latest attempts, marking evidence stale after source changes."""
    manifest = _read(root)
    try:
        current = scene_fingerprint(root)
        manifest["current_fingerprint"] = current["sha256"]
    except (OSError, ValueError) as exc:
        manifest["current_fingerprint"] = None
        manifest["fingerprint_error"] = str(exc)
    for attempt in manifest["operations"].values():
        previous = attempt.get("fingerprint")
        attempt["stale"] = not previous or previous["sha256"] != manifest["current_fingerprint"]
    return manifest

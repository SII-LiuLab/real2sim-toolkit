"""Explicit, versioned public memory for one reconstruction session.

The caller supplies public goals, constraints, decisions, facts or failed
approaches. This module never extracts restrictions from chat or saves private
reasoning. Evidence hashes and optional focus/scene dependencies are checked at
recall time; stable session decisions do not implicitly depend on scene.py.

The append-only journal is authoritative. A process lock serializes readers and
writers, and an atomic materialized JSON index is rebuilt from committed events.
``forget_memory`` writes a tombstone; it is not erasure of the audit history.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from urllib.parse import parse_qsl, urlsplit

from .io import read_json
from .session import _atomic_json, scene_fingerprint


KINDS = frozenset({"goal", "constraint", "decision", "fact", "failed_approach"})
SOURCES = frozenset({"user", "observed", "inferred"})
MAX_CONTENT_BYTES = 8192
_SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[A-Z0-9]{16})\b|"
    r"(?i:authorization\s*:\s*bearer\s+\S{12,})|(?i:<(?:analysis|think)\b))"
)
_PRIVATE_KEYS = {"password", "api_key", "access_token", "refresh_token", "private_key",
                 "credentials", "authorization", "secret", "token", "private_reasoning",
                 "chain_of_thought", "hidden_reasoning"}


class MemoryConflictError(ValueError):
    """A key already has a newer revision; explicitly supersede its exact ID."""


class MemoryStoreError(ValueError):
    """The memory journal is invalid; fail closed instead of losing history."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _public(value, *, limit, label):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error
    if len(encoded.encode("utf-8")) > limit:
        raise ValueError(f"{label} exceeds {limit} UTF-8 bytes")

    def inspect(item):
        if isinstance(item, str) and _SECRET.search(item):
            raise ValueError(f"{label} must not contain credentials or private reasoning")
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower().replace("-", "_") in _PRIVATE_KEYS and child:
                    raise ValueError(f"{label} must not contain credential/private-reasoning fields")
                inspect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)
    inspect(value)
    return json.loads(encoded)


def _inside(root, value):
    path = Path(value)
    lexical = Path(os.path.abspath(path if path.is_absolute() else root/path))
    try:
        relative = lexical.relative_to(root)
        lexical.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("Memory paths and symlinks must remain inside the session") from error
    return lexical, relative.as_posix()


def _session(value):
    root = Path(value).expanduser().resolve()
    if not (root/"session.json").is_file():
        raise ValueError("Initialize a reconstruction session before using its memory")
    manifest = read_json(root/"session.json")
    if not isinstance(manifest, dict):
        raise ValueError("Invalid reconstruction session manifest")
    return root


@contextmanager
def _locked(root):
    # Imported lazily: memory can still be imported by ordinary CPU tooling.
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("Session memory requires POSIX file locking") from error
    directory, _ = _inside(root, ".agent/memory")
    if directory.is_symlink() or (root/".agent").is_symlink():
        raise ValueError("Memory control directories must not be symlinks")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("lock", "events.jsonl", "state.json"):
        path, _ = _inside(root, directory/name)
        if path.is_symlink():
            raise ValueError("Memory control files must not be symlinks")
    descriptor = os.open(directory/"lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield directory
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load(directory):
    current = {}
    sequence = 0
    journal = directory/"events.jsonl"
    if journal.exists():
        with journal.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    if not line.endswith("\n"):
                        raise ValueError("Incomplete trailing event")
                    event = json.loads(line)
                    record = event["record"]
                    previous = current.get(record["key"])
                    expected_id = previous["id"] if previous else None
                    if (event.get("schema_version") != 1 or event["sequence"] != sequence+1
                            or event["operation"] not in {"put", "forget"}
                            or record["kind"] not in KINDS or record["scope"] not in {"session", "focus"}
                            or record["status"] not in {"active", "forgotten"}
                            or record["supersedes"] != expected_id
                            or record["revision"] != (previous["revision"]+1 if previous else 1)):
                        raise ValueError("Invalid event revision chain")
                    current[record["key"]] = record
                    sequence += 1
                except (ValueError, TypeError, KeyError) as error:
                    raise MemoryStoreError(f"Invalid memory journal at line {line_number}: {error}") from error
    index = directory/"state.json"
    if index.is_file():
        try:
            previous = read_json(index)
        except (OSError, ValueError):
            previous = {}  # A damaged derived index is recoverable from the log.
        if isinstance(previous, dict) and isinstance(previous.get("sequence"), int) and previous["sequence"] > sequence:
            raise MemoryStoreError("Memory journal is missing/truncated relative to its materialized index")
    return {"schema_version": 1, "sequence": sequence, "current": current}


def _commit(directory, state, operation, record):
    event = {"schema_version": 1, "sequence": state["sequence"]+1,
             "operation": operation, "record": record}
    journal = directory/"events.jsonl"
    descriptor = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        payload = (json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    state["sequence"] = event["sequence"]
    state["current"][record["key"]] = record
    state["updated_at"] = _now()
    _atomic_json(directory/"state.json", state)
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w][\w.:-]{0,95}", value):
        raise ValueError("Memory key must be 1–96 letters/digits or _, ., :, -")
    return value


def _source(value):
    result = {"type": value} if isinstance(value, str) else value
    result = _public(result, limit=2048, label="source")
    if not isinstance(result, dict) or result.get("type") not in SOURCES:
        raise ValueError("source must be user/observed/inferred or an object with that type")
    if "depends_on_scene" in result and not isinstance(result["depends_on_scene"], bool):
        raise ValueError("source.depends_on_scene must be boolean")
    return result


def _focus(root):
    path, _ = _inside(root, ".agent/focus.json")
    if not path.is_file():
        return None
    data = read_json(path)
    revision = data.get("revision") if isinstance(data, dict) else None
    if isinstance(revision, bool) or not isinstance(revision, (str, int)) or revision == "":
        raise ValueError("Current focus must contain a nonempty string/integer revision")
    return revision


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(root, values):
    if values is None:
        return []
    if isinstance(values, (str, Path, dict)):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) > 16:
        raise ValueError("evidence must contain at most 16 session file paths or URL objects")
    result = []
    for value in values:
        item = {"path": str(value)} if isinstance(value, (str, Path)) else value
        item = _public(item, limit=2048, label="evidence")
        if not isinstance(item, dict) or ("path" in item) == ("url" in item):
            raise ValueError("Each evidence item needs exactly one path or url")
        if "path" in item:
            path, relative = _inside(root, item["path"])
            if not path.is_file():
                raise ValueError(f"Memory evidence must be an existing session file: {relative}")
            if relative.startswith(".agent/memory/"):
                raise ValueError("Memory cannot use its own changing store as evidence")
            result.append({"path": relative, "sha256": _hash(path), "size": path.stat().st_size,
                           **({"label": item["label"]} if "label" in item else {})})
        else:
            url = urlsplit(item["url"])
            if url.scheme not in {"https", "http"} or not url.netloc or url.username or url.password:
                raise ValueError("Evidence URLs must be public HTTP(S) URLs without credentials")
            if any(key.lower() in {"token", "api_key", "access_token", "password", "signature"}
                   for key, _ in parse_qsl(url.query)):
                raise ValueError("Do not store credential-bearing evidence URLs")
            result.append(item)
    return result


def put_memory(session, *, key, kind, content, source, evidence=None, tags=None,
               supersedes=None, scope="session", focus_revision=None):
    """Store explicit public input, or replace a key with its exact current ID.

    ``source`` is user/observed/inferred or ``{"type": ..., "note": ...}``.
    Non-user facts/failed approaches may explicitly set ``depends_on_scene``;
    constraints/decisions never acquire that dependency implicitly. Focus-scoped
    memories require a current ``.agent/focus.json`` revision. File evidence is
    always explicit, portable and content-hashed; URLs are retained, not fetched.
    """
    root = _session(session)
    key = _key(key)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    if scope not in {"session", "focus"}:
        raise ValueError("scope must be session or focus")
    content = _public(content, limit=MAX_CONTENT_BYTES, label="content")
    if content is None or content == "":
        raise ValueError("content must be nonempty explicit public data")
    source = _source(source)
    if not isinstance(tags or [], (list, tuple)) or len(tags or []) > 16:
        raise ValueError("tags must contain at most 16 strings")
    if any(not isinstance(tag, str) or not tag or len(tag.encode("utf-8")) > 64 for tag in (tags or [])):
        raise ValueError("Each tag must be a nonempty string of at most 64 UTF-8 bytes")
    tags = list(dict.fromkeys(tags or []))
    _public(tags, limit=2048, label="tags")
    with _locked(root) as directory:
        state = _load(directory)
        previous = state["current"].get(key)
        if previous is not None and supersedes != previous["id"]:
            raise MemoryConflictError(f"Memory key {key!r} already exists; supersedes={previous['id']} is required")
        if previous is None and supersedes is not None:
            raise MemoryConflictError("Cannot supersede a nonexistent memory key")
        if scope == "focus":
            current_focus = _focus(root)
            if current_focus is None:
                raise ValueError("Set an explicit session focus before storing focus-scoped memory")
            if focus_revision is not None and focus_revision != current_focus:
                raise MemoryConflictError("focus_revision does not match the current session focus")
            focus_revision = current_focus
        elif focus_revision is not None:
            raise ValueError("focus_revision applies only to scope=focus")
        scene_dependency = None
        if source.get("depends_on_scene"):
            if kind not in {"fact", "failed_approach"} or source["type"] == "user":
                raise ValueError("Only observed/inferred facts or failed approaches can depend on the scene")
            scene_dependency = {"sha256": scene_fingerprint(root)["sha256"]}
        record = {"id": uuid.uuid4().hex, "key": key,
            "revision": previous["revision"]+1 if previous else 1,
            "kind": kind, "content": content, "source": source,
            "evidence": _evidence(root, evidence), "tags": tags,
            "scope": scope, "focus_revision": focus_revision,
            "supersedes": supersedes, "created_at": _now(), "status": "active",
            "scene_dependency": scene_dependency}
        _commit(directory, state, "put", record)
    return record


def _recall_record(root, stored, current_focus, scene_cache):
    record = json.loads(json.dumps(stored))
    reasons = []
    checks = []
    if record["status"] == "forgotten":
        reasons.append("forgotten")
    if record["scope"] == "focus" and record["focus_revision"] != current_focus:
        reasons.append("focus_changed")
    for item in record["evidence"]:
        if "path" not in item:
            continue
        check = {"path": item["path"], "expected_sha256": item["sha256"]}
        try:
            path, _ = _inside(root, item["path"])
            current = _hash(path) if path.is_file() else None
            check.update(sha256=current, status="current" if current == item["sha256"] else "changed" if current else "missing")
        except (OSError, ValueError) as error:
            check.update(sha256=None, status="unverifiable", error=str(error))
        if check["status"] != "current":
            reasons.append(f"evidence_{check['status']}:{item['path']}")
        checks.append(check)
    dependency = record.get("scene_dependency")
    if dependency:
        if not scene_cache:
            try:
                scene_cache["sha256"] = scene_fingerprint(root)["sha256"]
            except (OSError, ValueError):
                scene_cache["sha256"] = None
        if dependency["sha256"] != scene_cache["sha256"]:
            reasons.append("scene_changed" if scene_cache["sha256"] else "scene_unverifiable")
    record.update(stale=bool(reasons), stale_reasons=reasons, evidence_status=checks)
    return record


def recall_memory(session, *, query=None, kind=None, limit=12, include_stale=False, priority_kinds=None):
    """Return latest records, checking evidence/focus dependencies on each read."""
    root = _session(session)
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if not isinstance(include_stale, bool):
        raise ValueError("include_stale must be boolean")
    if priority_kinds is not None:
        if (not isinstance(priority_kinds, (list, tuple)) or
                any(value not in KINDS for value in priority_kinds) or
                len(set(priority_kinds)) != len(priority_kinds)):
            raise ValueError("priority_kinds must be a sequence of distinct supported memory kinds")
    priorities = {name: len(priority_kinds)-index for index, name in enumerate(priority_kinds or [])}
    if query is not None and (not isinstance(query, str) or len(query.encode("utf-8")) > 2048):
        raise ValueError("query must be a string of at most 2048 UTF-8 bytes")
    tokens = re.findall(r"\w+", (query or "").casefold())
    with _locked(root) as directory:
        state = _load(directory)
        current_focus = _focus(root)
        scene_cache = {}
        matches = []
        for stored in state["current"].values():
            if kind is not None and stored["kind"] != kind:
                continue
            text = json.dumps({key: stored[key] for key in ("key", "kind", "content", "source", "tags")},
                              ensure_ascii=False).casefold()
            if tokens and not all(token in text for token in tokens):
                continue
            record = _recall_record(root, stored, current_focus, scene_cache)
            if record["stale"] and not include_stale:
                continue
            score = sum(text.count(token) for token in tokens)
            matches.append((priorities.get(record["kind"], 0), score, record["created_at"], record))
        matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        # Repair a stale/missing materialized index from the authoritative log.
        _atomic_json(directory/"state.json", state)
        return [record for _, _, _, record in matches[:limit]]


def forget_memory(session, key):
    """Append an explicit tombstone; audit history remains available on disk."""
    root = _session(session)
    key = _key(key)
    with _locked(root) as directory:
        state = _load(directory)
        previous = state["current"].get(key)
        if previous is None:
            raise ValueError(f"Unknown memory key {key!r}")
        if previous["status"] == "forgotten":
            return previous
        record = {**previous, "id": uuid.uuid4().hex, "revision": previous["revision"]+1,
            "content": None, "source": {"type": "observed", "note": "Explicit forget_memory API call"},
            "evidence": [], "supersedes": previous["id"], "created_at": _now(),
            "status": "forgotten", "scene_dependency": None}
        _commit(directory, state, "forget", record)
        return record

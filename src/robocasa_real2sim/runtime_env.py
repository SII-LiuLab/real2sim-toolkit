"""Inspect the pinned Studio environment without importing simulation packages.

Only Python's standard library is used. ``ready`` means the selected profile's
package metadata and selected bundled sources match the lock; it does not establish GPU,
driver, solver, or renderer functionality. A strict Studio launch should call
``assert_studio_runtime(renderer=...)`` before importing RoboSimStudio/Newton.
Upstream identity records origin, not an unchanged upstream directory layout.
The current runtime may omit upstream test/demo material or relocate domain packs.
"""

import ast
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from .studio_runtime_lock import STUDIO_RUNTIME_LOCK


_SOURCE_IGNORES = {".git", ".venv", "venv", "__pycache__", ".cache", ".pytest_cache",
                   ".ruff_cache", ".mypy_cache", "outputs", "dist", "node_modules"}


class StudioRuntimeError(RuntimeError):
    """The selected runtime differs from the portable lock; see ``report``."""

    def __init__(self, report):
        self.report = report
        details = "; ".join(issue["message"] for issue in report["mismatches"])
        super().__init__(f"Studio runtime lock mismatch ({report['lock_digest']}): {details}")


def _git(path, *args):
    try:
        environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        process = subprocess.run(
            ["git", "-C", str(path), *args],
            text=True, capture_output=True, timeout=5, check=False, env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"
    if process.returncode:
        return None, "Git could not inspect the installed source checkout"
    return process.stdout.strip(), None


def _provenance(distribution):
    """PEP 610 records provenance; editable installs require a live Git check."""
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return {"kind": "index_or_unknown", "editable": False, "revision": None,
                "dirty": None}
    try:
        direct = json.loads(raw)
    except (TypeError, ValueError):
        return {"kind": "invalid_direct_url", "editable": None, "revision": None,
                "dirty": None, "verification_error": "Invalid direct_url.json"}
    if not isinstance(direct, dict):
        return {"kind": "invalid_direct_url", "editable": None, "revision": None,
                "dirty": None, "verification_error": "direct_url.json is not an object"}
    directory = direct.get("dir_info") or {}
    vcs = direct.get("vcs_info") or {}
    if not isinstance(directory, dict) or not isinstance(vcs, dict):
        return {"kind": "invalid_direct_url", "editable": None, "revision": None,
                "dirty": None, "verification_error": "Invalid PEP 610 directory/VCS metadata"}
    editable = bool(directory.get("editable", False))
    result = {"kind": "editable" if editable else "vcs" if vcs else "archive_or_directory",
              "editable": editable, "revision": vcs.get("commit_id"), "dirty": None}
    if not editable:
        # A VCS wheel's PEP 610 commit is retained even without a source checkout.
        # An archive/local directory with no commit cannot prove a Git lock.
        return result
    try:
        url = urlsplit(str(direct.get("url", "")))
        if url.scheme != "file" or url.netloc not in {"", "localhost"}:
            raise ValueError("Editable source is not a local file URL")
        source = Path(unquote(url.path)).resolve()
        result["path"] = str(source)
    except (OSError, ValueError) as error:
        result["verification_error"] = str(error)
        return result
    revision, error = _git(source, "rev-parse", "HEAD")
    if error:
        result["verification_error"] = error
        return result
    result["revision"] = revision
    dirty, error = _git(source, "status", "--porcelain=v1", "--untracked-files=normal")
    if error:
        result["verification_error"] = error
    else:
        result["dirty"] = bool(dirty)
    return result


def _embedded_project(expected):
    """Resolve the shared source project from this harness, never cwd."""
    package = Path(__file__).resolve().parent
    if package.parent.name != "src":
        return None
    project = package.parent.parent
    candidate = project/expected["local_path"]
    return project if candidate.is_dir() else None


def _active_module_path(name):
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
    else:
        spec = find_spec(name)
        origin = spec.origin if spec is not None else None
    if not origin or origin in {"built-in", "frozen"}:
        raise ValueError(f"Cannot locate the active {name} module without importing it")
    return Path(origin).resolve()


def _source_fingerprint(paths, scope):
    """Aggregate content evidence without embedding a potentially large file list."""
    entries = []
    total = 0
    for relative, path in sorted(paths, key=lambda item: item[0]):
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
                total += len(chunk)
        entries.append({"path": relative, "sha256": digest.hexdigest()})
    digest = sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "file_count": len(entries), "total_bytes": total,
            "scope": scope, "algorithm": "sha256-canonical-path-content-list"}


def _checkout_files(root):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _SOURCE_IGNORES
                         and not (Path(directory) == root and name == "build"))
        for name in dirs:
            if (Path(directory)/name).is_symlink():
                raise ValueError("Bundled source fingerprint cannot follow directory symlinks")
        for name in names:
            if name.endswith((".pyc", ".pyo", ".tmp")):
                continue
            path = Path(directory)/name
            path.resolve().relative_to(root)
            if path.is_file():
                yield path.relative_to(root).as_posix(), path


def _embedded_provenance(distribution, expected, source, issue):
    """Verify RSS and harness belong to one root distribution without importing RSS."""
    result = {**source, "upstream_provenance": deepcopy(expected["upstream_provenance"]),
              "source_policy": "RSS embedded in the root distribution; upstream identity records origin, current source fingerprint is informational"}
    try:
        module = expected["module"]
        active = _active_module_path(module)
        harness_active = _active_module_path("robocasa_real2sim")
        result["active_module"] = str(active)
        result["harness_active_module"] = str(harness_active)
        result["distribution"] = expected["distribution"]
        result["distribution_location"] = str(Path(distribution.locate_file("")).resolve())
        project = _embedded_project(expected)
        if source.get("editable"):
            result["kind"] = "embedded_editable"
            if project is None:
                issue("embedded_source_missing", "Cannot locate embedded RSS in this harness source project")
                return result
            result["source_project"] = str(project)
            if (source.get("path") != str(project)
                    or active != project/expected["local_path"]/"__init__.py"
                    or harness_active != project/"src/robocasa_real2sim/__init__.py"):
                issue("external_shadowing", "Active RSS/harness does not belong to the root editable distribution; reinstall the root project and remove external import paths")
                return result
        else:
            result["kind"] = "embedded_wheel"
            installed = Path(distribution.locate_file(f"{module}/__init__.py")).resolve()
            installed_harness = Path(distribution.locate_file("robocasa_real2sim/__init__.py")).resolve()
            files = distribution.files
            names = {str(path).replace("\\", "/") for path in files or []}
            if active != installed or harness_active != installed_harness:
                issue("external_shadowing", "Active RSS/harness does not match the root distribution files")
                return result
            required_modules = ["robocasa_real2sim", *expected["module_packages"]]
            if any(f"{name}/__init__.py" not in names for name in required_modules):
                issue("embedded_source_unverifiable", "Root distribution RECORD does not contain all embedded runtime packages", "unverifiable")
                return result
        marker_path = active.parent/expected["origin_marker"]
        if not marker_path.is_file():
            issue("bundled_origin_unverifiable", "RoboSimStudio lacks the real2sim embedded runtime origin marker", "unverifiable")
            return result
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        upstream = expected["upstream_provenance"]
        required = {"schema_version": 1, "project": upstream["project"],
                    "upstream_url": upstream["url"], "upstream_revision": upstream["revision"],
                    "original_source_digest": upstream["original_source_digest"],
                    "integration": expected["integration"]}
        if not isinstance(marker, dict) or any(marker.get(key) != value for key, value in required.items()):
            issue("bundled_origin", "RoboSimStudio real2sim embedded runtime marker differs from recorded upstream provenance")
            return result
        result["origin_marker"] = required
        if "slim_profile" in marker:
            result["slim_profile"] = deepcopy(marker["slim_profile"])
        if "runtime_layout" in marker:
            result["runtime_layout"] = deepcopy(marker["runtime_layout"])
        version_tree = ast.parse((active.parent/"_version.py").read_text(encoding="utf-8"))
        versions = [node.value.value for node in version_tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
        if len(versions) != 1:
            raise ValueError("Embedded RSS version must be a single literal __version__ assignment")
        result["component_version"] = versions[0]
        if source.get("editable"):
            paths = []
            for relative_root in expected["source_paths"]:
                source_root = project/relative_root
                if not source_root.is_dir():
                    raise ValueError(f"Missing embedded source directory: {relative_root}")
                paths.extend((f"{relative_root}/{relative}", path)
                             for relative, path in _checkout_files(source_root))
            scope = "embedded RSS runtime packages and third_party provenance only; generated files excluded"
        else:
            marker_relative = f"{module}/{expected['origin_marker']}"
            if marker_relative not in names:
                issue("bundled_origin_unverifiable", "Origin marker is not part of the installed distribution RECORD", "unverifiable")
                return result
            paths = []
            for entry in files:
                relative = str(entry).replace("\\", "/")
                parts = Path(relative).parts
                embedded = parts and (parts[0] in expected["module_packages"]
                                      or "/third_party/robosimstudio/" in "/"+relative)
                if (not embedded or ".." in parts
                        or any(part in _SOURCE_IGNORES for part in parts)
                        or relative.endswith((".pyc", ".pyo"))):
                    continue
                path = Path(distribution.locate_file(entry))
                if path.is_file():
                    paths.append((relative, path))
            scope = "embedded RSS runtime files in the single root distribution"
        result["source_fingerprint"] = _source_fingerprint(paths, scope)
        # revision/dirty describe the current parent repository when editable;
        # they are information, not an upstream-content equality requirement.
    except (OSError, ValueError, ImportError, AttributeError, SyntaxError) as error:
        result["verification_error"] = f"{type(error).__name__}: {error}"
        issue("embedded_source_unverifiable", f"Cannot verify embedded RoboSimStudio: {error}", "unverifiable")
    return result


def _package(name, expected, group):
    record = {"name": name, "expected": deepcopy(expected), "installed_version": None,
              "status": "ready", "provenance": None, "issues": []}

    def issue(kind, message, status="mismatched"):
        record["issues"].append({"kind": kind, "package": name, "group": group,
                                 "status": status, "message": message})

    try:
        distribution = metadata.distribution(expected.get("distribution", name))
    except metadata.PackageNotFoundError:
        record["status"] = "missing"
        issue("missing", f"{expected.get('distribution', name)} is missing; expected {expected.get('distribution_version', expected['version'])}", "missing")
        return record
    record["installed_version"] = distribution.version
    record["distribution"] = expected.get("distribution", name)
    record["installed_distribution_version"] = distribution.version
    distribution_version = expected.get("distribution_version", expected["version"])
    if distribution.version != distribution_version:
        issue("version", f"{record['distribution']}=={distribution.version}; expected =={distribution_version}")
    try:
        source = _provenance(distribution)
    except (OSError, ValueError, TypeError) as error:
        source = {"kind": "unverifiable", "editable": None, "revision": None,
                  "dirty": None, "verification_error": f"{type(error).__name__}: {error}"}
    record["provenance"] = source
    if expected.get("source") == "embedded":
        source = _embedded_provenance(distribution, expected, source, issue)
        record["provenance"] = source
        record["installed_version"] = source.get("component_version")
        if source.get("component_version") and source["component_version"] != expected["version"]:
            issue("component_version", f"Embedded {name}=={source['component_version']}; expected =={expected['version']}")
    revision = expected.get("git_revision")
    if revision:
        if source["revision"] is None:
            issue("revision_unverifiable", f"{name} Git revision cannot be verified; expected {revision}",
                  "unverifiable")
        elif str(source["revision"]).lower() != revision.lower():
            issue("revision", f"{name} Git revision is {source['revision']}; expected {revision}")
        if source.get("editable") and expected.get("editable_must_be_clean"):
            if source["dirty"] is None:
                issue("dirty_unverifiable", f"{name} editable checkout cleanliness cannot be verified",
                      "unverifiable")
            elif source["dirty"]:
                issue("dirty", f"{name} editable checkout is dirty; the runtime lock requires clean source")
    if record["issues"]:
        record["status"] = ("mismatched" if any(i["status"] == "mismatched" for i in record["issues"])
                            else "unverifiable")
    return record


def runtime_report(renderer="tiled"):
    """Return JSON-serializable lock/provenance diagnostics; never import GPU code.

    Required packages and Python 3.12 are always checked. RTX-only packages are
    reported separately and become blocking only for ``renderer='rtx'``.
    ``auto`` checks the required profile; check the resolved renderer again if
    the caller later chooses RTX. RSS and the harness must resolve to one root
    distribution, editable or wheel. Its repository may be dirty: source changes are
    fingerprinted, while the retained upstream revision describes provenance.
    Other explicitly pinned Git dependencies retain their original policy.
    """
    if renderer not in {"tiled", "rtx", "auto"}:
        raise ValueError("renderer must be 'tiled', 'rtx', or 'auto'")
    lock = deepcopy(STUDIO_RUNTIME_LOCK)
    digest = sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    actual_python = {"implementation": platform.python_implementation(),
                     "version": platform.python_version(), "major": sys.version_info.major,
                     "minor": sys.version_info.minor}
    expected_python = lock["python"]
    required_issues = []
    if any(actual_python[k] != expected_python[k] for k in ("implementation", "major", "minor")):
        required_issues.append({"kind": "python", "group": "required", "status": "mismatched",
            "message": f"Python {actual_python['implementation']} {actual_python['version']}; expected CPython 3.12.x"})
    groups = {}
    for group in ("required", "rtx_optional"):
        groups[group] = {name: _package(name, expected, group)
                         for name, expected in lock[group].items()}
    required_issues.extend(issue for item in groups["required"].values() for issue in item["issues"])
    rtx_issues = [issue for item in groups["rtx_optional"].values() for issue in item["issues"]]
    blocking = required_issues + (rtx_issues if renderer == "rtx" else [])
    return {
        "schema_version": 1, "status": "mismatch" if blocking else "ready",
        "passed": not blocking, "renderer": renderer,
        "lock_resource": "robocasa_real2sim.studio_runtime_lock:STUDIO_RUNTIME_LOCK",
        "lock_digest_algorithm": "sha256-canonical-json", "lock_digest": digest,
        "expected": lock, "python": actual_python,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "installed_versions": {name: item["installed_version"]
                               for group in groups.values() for name, item in group.items()},
        "packages": groups, "required_mismatches": required_issues,
        "rtx_optional_mismatches": rtx_issues, "mismatches": blocking,
        "required_status": "mismatch" if required_issues else "ready",
        "rtx_optional_status": "mismatch" if rtx_issues else "ready",
        "rtx_optional_required": renderer == "rtx",
        "checks_scope": "Distribution/component versions, active embedded RSS and harness routing/origin and content fingerprint; other Git dependency provenance; no GPU imports or device initialization",
        "gpu_driver_verified": False, "simulation_verified": False, "rendering_verified": False,
    }


def assert_studio_runtime(renderer="tiled"):
    """Return the matching report or raise ``StudioRuntimeError(report)``."""
    report = runtime_report(renderer=renderer)
    if report["status"] != "ready":
        raise StudioRuntimeError(report)
    return report

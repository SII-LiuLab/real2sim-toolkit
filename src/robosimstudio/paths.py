# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit scene assets without shipping a catalogue of robot models.

A relative path is resolved against the scene's working directory. The harness
loads and runs each native scene from its own session directory. Absolute paths
are accepted; portable sessions should keep geometry and its dependencies
inside the session. ``ROBOSIMSTUDIO_ASSETS`` optionally supplies an explicit
shared base directory instead of the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from robosimstudio.specs import AssetSpec

__all__ = ["asset_root", "resolve_asset", "resolve_path"]

_ENV_VAR = "ROBOSIMSTUDIO_ASSETS"


def _candidate_roots() -> list[Path]:
    override = os.environ.get(_ENV_VAR)
    return [Path(override).expanduser().resolve()] if override else [Path.cwd().resolve()]


def asset_root() -> Path:
    """The explicit shared asset base, or the current scene directory."""
    return _candidate_roots()[0]


def resolve_path(path: str | Path) -> Path:
    """Resolve an explicit absolute or scene-relative geometry path."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = asset_root() / candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"Scene asset {path!s} not found at {candidate}. No built-in models are shipped. "
            "Copy geometry and its dependencies into the session and reference their explicit paths, "
            f"or set {_ENV_VAR} to a shared asset directory."
        )
    return candidate.resolve()


def resolve_asset(spec: AssetSpec) -> Path:
    """Resolve an asset spec using the scene's explicit asset paths."""
    return resolve_path(spec.path)

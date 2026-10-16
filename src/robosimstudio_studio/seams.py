# SPDX-License-Identifier: Apache-2.0
"""Clear compatibility errors for the removed benchmark and policy interfaces.

The native page accepts a compiled Sim and retains its camera, recording and
manipulation controls. Benchmark task loading and policy panels are absent.
These small shims keep older imports from failing with an unrelated traceback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["ADDONS", "WITHHELD", "NotInThisRelease", "benchmark", "benchmark_or_none", "lazy_names"]

WITHHELD = ("robosimstudio_bench", "robosimstudio_policy_agent", "robosimstudio_repro")
ADDONS = ("bench_session", "bench_panels", "goal_panel", "fleet_session", "fleet_view")


class NotInThisRelease(ImportError):
    """A removed task or policy interface was explicitly requested."""


def benchmark(what: str):
    """Reject legacy task/policy use before a page or server is opened."""
    raise NotInThisRelease(
        f"{what} is not included in this reconstruction runtime. "
        "Build an explicit scene with rss.Scene(...).build() and open its native asset UI."
    )


def benchmark_or_none() -> None:
    """The asset page has no benchmark goal decorations."""
    return None


def lazy_names(module: str, names: Mapping[str, str]) -> Callable[[str], Any]:
    """Keep a clear error for explicitly requested, removed panel names."""
    def __getattr__(name: str) -> Any:
        if name in names:
            return benchmark(f"{module}.{name}")
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    return __getattr__

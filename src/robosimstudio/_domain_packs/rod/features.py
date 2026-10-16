# SPDX-License-Identifier: Apache-2.0
"""What only a chain can carry: the arc-length ``point``, and the protocol a chain answers.

Registers the ``point`` feature kind (a position by normalised arc length).
Only the ``rod`` layout can carry it; the validator refuses one on a rigid body.
:data:`KIND_NAMES` is this pack's addition to
``assetlib.semantics.library.KIND_NAMES``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from robosimstudio.assetlib.semantics.kinds import FeatureKind, register_kind

__all__ = ["KIND_NAMES", "POINT", "RodParam"]


POINT = register_kind(
    FeatureKind(
        name="point",
        summary="A position along a 1-D body, by normalised arc length.",
        payload={"s": "arc length in [0, 1]"},
        contracts=(
            "arc length is normalised, so a reference survives a change of segment count",
            "material identity, not world position: the cable moves, its midpoint does not stop being its midpoint",
        ),
        substrates=("rod",),
        provenance="Free from the rod substrate's own parameterisation.",
    )
)

KIND_NAMES: tuple[str, ...] = ("point",)
"""Every feature kind this domain registers, sorted. Pinned by the same test as the library's."""


@runtime_checkable
class RodParam(Protocol):
    """A 1-D body addressed by normalised arc length in ``[0, 1]``."""

    def point(self, s: float) -> np.ndarray:
        """World position at normalised arc length ``s`` in ``[0, 1]`` [m]."""
        ...

    def segment(self, s: float) -> int:
        """Index of the segment containing normalised arc length ``s``."""
        ...

    def positions(self, body_q: np.ndarray) -> np.ndarray:
        """Every segment's world position ``[n, 3]``."""
        ...

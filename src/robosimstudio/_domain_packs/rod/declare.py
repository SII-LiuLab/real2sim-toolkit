# SPDX-License-Identifier: Apache-2.0
"""Declaring a 1-D body: the part a rod already knows, said once.

Builds an :class:`AssetRecord` for a rod: a ``point`` feature per landmark and
``sim=SimRequirements(domain="rod", contacts=contacts_with(...))``. A landmark is
a normalised arc length, nothing measured, so its provenance is stated for you
(:data:`DERIVED`). Anything measured (the wrap's anchor at s = 0.6361, the line's
straightness gate of 0.98, an axis the posts define) is passed in with its own
provenance. Straightness is a ``states=`` entry like any other.

    ROPE = Rod.from_length(
        name="tow_rope", length=2.0, radius=0.008,
        summary="A 2 m rope.", points={"end_a": 0.0, "end_b": 1.0},
        touches=("surface", "fixtures", "hands"),
    )

This builds a record and never registers one; the task registers it under its
own name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

__all__ = ["Rod"]

DERIVED = "The chain's own arc-length parameterisation; nothing measured."
"""Provenance for every landmark on a rod: the number is the definition."""


class Rod:
    """Constructors for a 1-D body, named for what you already have."""

    @staticmethod
    def from_length(
        *,
        name: str,
        length: float,
        radius: float,
        summary: str = "",
        points: Mapping[str, float | tuple[float, str]] = (),
        touches: Sequence[str] = ("surface", "fixtures", "hands"),
        tags: Sequence[str] = ("rod",),
        features: Sequence[FeatureSpec] = (),
        states: Sequence[StateSpec] = (),
        published: Sequence[PublishedState] = (),
        notes: str = "",
        provenance: str = "",
        self_contact: bool = False,
        build: Callable[[], Any] | None = None,
    ) -> AssetRecord:
        """A rod of ``length`` metres, as an :class:`AssetRecord`.

        Args:
            name: What the *task* calls it. Not registered here.
            length: Overall length [m], used only for the default summary; the
                geometry is the entity's, built from the task's spec.
            radius: Cross-section radius [m], likewise only for the summary.
            summary: One line for the catalog. Derived from ``length`` and
                ``radius`` when omitted.
            points: Landmark name -> normalised arc length, or
                ``(arc length, why it is named)``. Each becomes a ``point``
                feature whose provenance is :data:`DERIVED` plus the note.
            touches: Roles this rod contacts, passed to
                :func:`~robosimstudio.assetlib.sim.contacts_with`.
            features: Anything **measured** -- an axis, an anchor, a cavity.
                Stated normally, with its own provenance.
            states, published, notes, provenance, tags, self_contact: Passed
                through unchanged.
            build: The factory a document builds this rod through; see
                :attr:`~robosimstudio.assetlib.record.AssetRecord.build`.

        Raises:
            ValueError: If a landmark is outside ``[0, 1]``, or if ``points``
                and ``features`` name the same thing.
        """
        landmarks = dict(points or {})
        for landmark, entry in landmarks.items():
            s = entry[0] if isinstance(entry, tuple) else entry
            if not 0.0 <= float(s) <= 1.0:
                raise ValueError(
                    f"rod {name!r}: landmark {landmark!r} is at s={s}, outside [0, 1]. "
                    "Arc length is normalised so a reference survives a change of segment count."
                )
        clash = sorted(landmarks.keys() & {f.name for f in features})
        if clash:
            raise ValueError(f"rod {name!r}: {clash} declared both as a landmark and as a feature; state one")

        derived = tuple(
            FeatureSpec(
                name=landmark,
                kind="point",
                payload={"s": float(entry[0] if isinstance(entry, tuple) else entry)},
                provenance=f"{DERIVED} {entry[1]}" if isinstance(entry, tuple) else DERIVED,
            )
            for landmark, entry in landmarks.items()
        )
        return AssetRecord(
            name=name,
            summary=summary or f"A {length:.3g} m rod, {radius * 2000:.0f} mm thick.",
            tags=tuple(tags),
            features=(*derived, *features),
            states=tuple(states),
            published=tuple(published),
            sim=SimRequirements(
                domain="rod",
                contacts=contacts_with(*touches),
                self_contact=self_contact,
                notes=notes,
            ),
            provenance=provenance,
            build=build,
        )

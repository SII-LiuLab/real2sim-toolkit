# SPDX-License-Identifier: Apache-2.0
"""Constructors for a 1-D deformable: free functions that return a spec.

Beside :mod:`~robosimstudio.domains.rod.declare`, which builds the
:class:`~robosimstudio.assetlib.record.AssetRecord` that says what a rod *is*.
This builds the :class:`~robosimstudio.domains.rod.RodSpec` that puts one in a
scene.
"""

from __future__ import annotations

from robosimstudio.specs import Vec3

from .entity import RodSpec

__all__ = ["rod"]


def rod(
    *,
    name: str = "rod",
    length: float = 0.9,
    radius: float = 0.011,
    at: Vec3 = (0.0, 0.0, 0.0),
    along: Vec3 = (0.0, 1.0, 0.0),
    segments: int = 28,
    **kwargs,
) -> RodSpec:
    """A capsule chain on cable joints: a cable, a rope, a suture, a hose.

    Args:
        length: End to end [m].
        radius: Capsule radius [m]. Segment spacing must stay above ``2 *
            radius``, so a short dense rod is the combination to watch.
        at: The **first** end (not the middle), in the work-surface frame, so
            ``z = radius`` rests on the bench. The chain runs ``length`` along
            ``along`` from here; to centre it, offset by half its length, as
            ``examples/03_rope/00_cable.py`` does with
            ``at=(x, -length / 2, radius + 0.01)``.
        along: Direction the chain runs in. Normalised for you.
        segments: Capsules. 28 is the shipped cable's; more is smoother and
            slower, and past roughly 40 the per-body contact buffer becomes the
            thing to watch -- see
            :data:`~robosimstudio.specs.physics.GENERIC`.

    A rod needs a soft solver: on MuJoCo the build fails inside MuJoCo's importer
    on a body mapping. ``GENERIC`` routes rods to ``vbd``.
    """
    return RodSpec(name=name, length=length, radius=radius, start=at, direction=along, segments=segments, **kwargs)

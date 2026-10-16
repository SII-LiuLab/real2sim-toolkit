# SPDX-License-Identifier: Apache-2.0
"""``Centerline``: a capability for anything that is a curve, whatever its domain.

A domain picks the solver; a capability says which measures and affordances a
thing gets. A cable simulated as a tetrahedral body is ``domain="soft"`` and one
simulated as a rolled shell is ``domain="cloth"``; either declares
:class:`Centerline` so ``winding_turns``, ``straightness`` and the grasp latch
still reach it. Implement :meth:`Centerline.centerline`; length, sag, hold and
the rod measures derive from it. The two implementations are
:class:`~robosimstudio.domains.rod.entity.EntityRod` (capsule poses from
``body_q``) and :class:`~robosimstudio.domains.rod.superdex.EntityRodSuperdex`
(nodes from ``particle_q``).

:func:`~robosimstudio.domains.capabilities.register_capability` publishes it as
``robosimstudio.domains.capabilities.Centerline``, so an entity in another domain
declares it without naming this pack.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from robosimstudio.domains.capabilities import register_capability

__all__ = ["Centerline"]


@register_capability
class Centerline:
    """Ordered node positions at an instant; everything else derives from them."""

    forms: ClassVar[tuple[str, ...]] = ("points",)
    """An ordered centerline, which is the whole content of this layout."""

    layouts: ClassVar[tuple[str, ...]] = ("rod",)
    """A curve reads as ``rod`` whatever its domain, so a soft-domain cable meets rod measures."""

    def centerline(self, snapshot: Any) -> np.ndarray:
        """``(n, 3)`` node positions, in order tip to tip, **at this instant**.

        Pass the instant: the model's own array is the build-time layout and
        never moves, so reading it reports a fallen cable where it was declared.
        """
        raise NotImplementedError(f"{type(self).__name__} declares Centerline and must implement centerline()")

    def substrate_points(self, snapshot: Any) -> np.ndarray | None:
        """This curve's ordered points, for any measure over the ``rod`` layout.

        Derived from :meth:`centerline`.
        """
        points = self.centerline(snapshot)
        if points is None or not len(points):
            return None
        return np.asarray(points, dtype=np.float64)

    # ------------------------------------------------------------- holding
    HOLD_REACH = 0.05
    """How near a hand's tool-centre point a node must be to be taken [m]."""

    def hold(
        self, on: bool, state: Any = None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Take the curve in ``arm``'s jaws so it rides the hand, or let it go.

        A capsule chain holds with the body latch (a ``rigid_force`` kernel); a
        particle rod holds with its engine's node pin, carried on the hand every
        substep.

        Args:
            on: Take, or let go.
            state: The instant, for the picker. Unused when letting go.
            arm: Which hand.
            slot: Which of two holds -- ``0`` the working hand, ``1`` an anchor.
            at: A node (or segment) index along the curve to take, instead of
                the one nearest the jaws. For a caller that already knows.

        Returns:
            The index taken along the curve, or ``-1`` -- let go, or nothing
            within :attr:`HOLD_REACH` of the jaws.
        """
        raise NotImplementedError(f"{type(self).__name__} declares Centerline and must implement hold()")

    def hand_tcp(self, arm: str, body_q: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """``(grip body, hand position, hand quaternion xyzw)`` for ``arm``, off the embodiment.

        Read from the hand's own declaration -- ``tcp_offset`` on the gripper,
        ``ee_body`` on the arm -- so no implementation carries a copy of how
        a hand is found. Raises by name on a scene with no such arm.
        """
        embodiment = getattr(self, "embodiment", None)
        if embodiment is None:
            raise LookupError(
                f"{type(self).__name__} {getattr(self, 'name', '?')!r}: no robot in this scene to hold with"
            )
        arms = embodiment.arms
        if arm not in arms:
            raise LookupError(f"no arm {arm!r}; this robot has {sorted(arms)}")
        grip = int(arms[arm].ee_body)
        hand = embodiment.hands.get(arm)
        offset = np.asarray(hand.tcp_offset if hand is not None else (0.0, 0.0, 0.0), dtype=np.float64)
        from robosimstudio.core.math import quat_to_mat

        q = np.asarray(body_q[grip], dtype=np.float64)
        pos = q[:3] + quat_to_mat(q[3:7]) @ offset
        return grip, pos, q[3:7]

    def nearest_to(self, points: np.ndarray, target: np.ndarray, *, at: int | None = None) -> int:
        """Which point along the curve to take: ``at``, or the nearest within reach, or ``-1``."""
        if at is not None:
            return int(at) if 0 <= int(at) < len(points) else -1
        if not len(points):
            return -1
        distances = np.linalg.norm(np.asarray(points, dtype=np.float64) - np.asarray(target, dtype=np.float64), axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= self.HOLD_REACH else -1

    def curve_states(self, snapshot: Any) -> dict[str, float]:
        """Arc length, sag and the sharpest bend.

        Not published by itself: an implementation folds these into its
        :meth:`~robosimstudio.scene.entity.Entity.published_states` to expose them.
        """
        points = self.centerline(snapshot)
        if points is None or len(points) < 2:
            return {}
        pts = np.asarray(points, dtype=np.float64)
        spans = np.diff(pts, axis=0)
        length = float(np.linalg.norm(spans, axis=1).sum())
        sag = float(pts[:, 2].max() - pts[:, 2].min())
        out = {"length": length, "sag": sag}
        if len(pts) > 2:
            # Menger curvature: the circumradius of each consecutive triple. The
            # smallest is the sharpest bend.
            a, b, c = pts[:-2], pts[1:-1], pts[2:]
            ab, bc, ca = b - a, c - b, a - c
            # |ab x bc| = 2 * area, so R = |ab||bc||ca| / (2 |ab x bc|).
            cross = np.linalg.norm(np.cross(ab, bc), axis=1)
            denom = 2.0 * np.maximum(cross, 1.0e-12)
            radius = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1) / denom
            out["min_bend_radius"] = float(radius[cross > 1.0e-12].min()) if np.any(cross > 1.0e-12) else float("inf")
        return out

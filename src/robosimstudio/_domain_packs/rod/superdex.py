# SPDX-License-Identifier: Apache-2.0
"""The same cable, simulated as a Cosserat rod by a foreign engine.

:class:`~robosimstudio.domains.rod.entity.EntityRod` builds a rod as
twenty-eight overlapping capsules on cable joints, integrated by VBD in maximal
coordinates. This builds the same cable as a **centerline**: one particle per
node, flagged inactive so no Newton solver touches them, integrated by
SuperDex's Cosserat rod, which carries a twist per node. Newton still owns the
memory, so the particle probe, the renderer, the read-back and the repro tape
work on it unchanged, as with the soft kind's sponge.

The arrangement picks one:

    ("rod", backend("vbd"))        -> EntityRod,         capsules on cable joints
    ("rod", backend("superdex"))   -> EntityRodSuperdex, a Cosserat centerline

and the spec is the same ``RodSpec`` either way.

A Cosserat rod wants ``EA``, ``EI`` and ``GJ``; ``RodSpec.stretch_ke`` and
``bend_ke`` are VBD penalty stiffnesses, not those. All three are derived from
``radius``, ``density``, ``RodSpec.young_modulus`` and ``poisson_ratio``;
``axial_stiffness`` overrides ``EA``.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.domains.capabilities import ParticleMixin, host_bridge
from robosimstudio.scene.entity import Entity
from robosimstudio.specs.physics import HostActor

from .centerline import Centerline
from .entity import RodSpec

if TYPE_CHECKING:
    from robosimstudio.core.pipeline import ScenePipeline

__all__ = ["EntityRodSuperdex"]

DEFAULT_FLAGS = int(inspect.signature(newton.ModelBuilder.add_particle).parameters["flags"].default)
"""The flags ``ModelBuilder.add_particle`` gives a particle when it is not told."""

INACTIVE = DEFAULT_FLAGS & ~int(newton.ParticleFlags.ACTIVE)
"""The default particle flags with ``ACTIVE`` cleared (a mask, not ``0``, in case
another flag is a default). Newton solvers and the soft-contact pass skip these
nodes; SuperDex writes them from the eager phase.
"""


class EntityRodSuperdex(ParticleMixin, Centerline, Entity):
    """One cable, as a centerline of inactive particles.

    Args:
        spec: The same :class:`~robosimstudio.domains.rod.entity.RodSpec` the
            Newton implementation takes.
        world: Which world.
    """

    spec: RodSpec
    domain = "rod"

    def __init__(self, spec: RodSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of this cable's centerline nodes, in order along it."""
        self.probe: ParticleProbe | None = None
        self.tube_radius = float(spec.radius)
        """What a renderer sweeps the centerline by: the cable's own radius."""
        self.rest: dict[str, np.ndarray] = {}
        self.bridge: Any = None
        """The engine's bridge, once the scene is ready; ``None`` on a Newton solver."""
        self._pins: dict[int, tuple[int, int]] = {}
        """Slot -> ``(pin id, global node)``. The engine carries the pin on the hand's body."""

    # -------------------------------------------------------------- phase 1
    def nodes(self) -> np.ndarray:
        """The centerline at rest, ``[segments + 1, 3]``, in world coordinates."""
        # The spec's rest polyline (stated, or straight), lifted into the world.
        return np.asarray(
            [self.placed((*node, 0.0, 0.0, 0.0, 1.0))[:3] for node in self.spec.rest_polyline()], dtype=np.float64
        )

    def stress_free_nodes(self) -> np.ndarray:
        """:attr:`RodSpec.stress_free` as nodes: the same legs as :meth:`nodes`, in the stated shape."""
        nodes = self.nodes()
        legs = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        shape = self.spec.stress_free
        if isinstance(shape, str):
            if shape != "straight":
                raise ValueError(f"rod {self.spec.name!r}: stress_free is 'straight' or a polyline, got {shape!r}")
            axis = (nodes[1] - nodes[0]) / legs[0]
            return nodes[0] + np.concatenate([[0.0], np.cumsum(legs)])[:, None] * axis[None, :]
        free = np.asarray(shape, dtype=np.float64).reshape(-1, 3)
        if free.shape != nodes.shape:
            raise ValueError(f"rod {self.spec.name!r}: stress_free has {len(free)} nodes, the rod {len(nodes)}")
        mismatch = np.abs(np.linalg.norm(np.diff(free, axis=0), axis=1) - legs).max()
        if mismatch > 1.0e-6:
            raise ValueError(f"rod {self.spec.name!r}: stress_free's legs differ from the rod's by {mismatch:.2e} m")
        return free

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        nodes = self.nodes()
        # Same total mass as the capsule chain, spread over the nodes.
        area = math.pi * float(spec.radius) ** 2
        node_mass = float(spec.density) * area * float(spec.length) / len(nodes)
        for point in nodes:
            builder.add_particle(
                pos=wp.vec3(*[float(v) for v in point]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=node_mass,
                radius=float(spec.radius),
                flags=INACTIVE,
            )

    def host_actor(self) -> HostActor:
        """What SuperDex needs beyond the nodes: the rod's matter, in its terms.

        ``EA``, ``EI`` and ``GJ`` for a circular section, from the modulus and
        the radius. Derived here because the model has no array to read them
        from (a tet body's Lame pair is read from ``tet_materials``).
        """
        spec = self.spec
        radius = float(spec.radius)
        young = float(spec.young_modulus)
        poisson = float(spec.poisson_ratio)
        area = math.pi * radius**2
        second_moment = math.pi * radius**4 / 4.0  # I, circular section
        shear = young / (2.0 * (1.0 + poisson))
        params: dict[str, Any] = {
            "radius": radius,
            "axial_stiffness": float(spec.axial_stiffness) if spec.axial_stiffness is not None else young * area,
            "flexural_stiffness": [young * second_moment, young * second_moment],
            # Polar second moment is 2I for a circle, so GJ = G * 2I.
            "torsional_stiffness": shear * 2.0 * second_moment,
            "linear_density": float(spec.density) * area,
            # rho * J per unit length, J = 2I for a circle.
            "linear_rotational_inertia": float(spec.density) * 2.0 * second_moment,
            "stiffness_damping": float(spec.stiffness_damping),
            "friction": float(spec.mu_cable) ** 0.5,
            "self_contact": bool(self.sim_requirements().self_contact),
            "contact_padding": float(spec.contact_gap),
            "contact_penalty": float(spec.contact_penalty),
            "skin": bool(spec.skin),
            "skin_segments": int(spec.skin_segments),
        }
        if spec.skin_samples:
            params["skin_samples"] = str(spec.skin_samples)
        if spec.stress_free is not None:
            params["stress_free"] = self.stress_free_nodes().tolist()
        for key in ("contact_threshold", "contact_smoothing", "friction_falloff"):
            value = getattr(spec, key)
            if value is not None:
                params[key] = float(value)
        count = int(spec.segments) + 1
        ends = {"start": 0, "end": count - 1}
        unknown = [a for a in spec.anchors if a not in ends]
        if unknown:
            raise ValueError(f"rod {spec.name!r}: anchors must name 'start' or 'end', got {unknown}")
        if spec.anchors:
            # A clamp: the end element's two nodes (position and tangent) and its
            # twist. See RodSpec.anchors for why not the end node alone.
            elements = {"start": 0, "end": count - 2}
            params["anchors"] = sorted({n for a in spec.anchors for n in (elements[a], elements[a] + 1)})
            params["anchor_twists"] = sorted({elements[a] for a in spec.anchors})
        welds = []
        if spec.joins:
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.nodes(), axis=0), axis=1))])
            for end, other, length in spec.joins:
                if end not in ends:
                    raise ValueError(f"rod {spec.name!r}: a join names 'start' or 'end', got {end!r}")
                from_end = arc if end == "start" else arc[-1] - arc
                nodes = np.flatnonzero(from_end <= float(length) + 1.0e-9)
                welds.append({"to": str(other), "nodes": [int(i) for i in nodes], "contact": False})
            params["welds"] = welds
        return HostActor(kind="rod", params=params)

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Bind the host bridge and join the shared settle record (no settling of its own).

        The record snapshots now and again after any later settle, so a reset
        never restores the particle array from before a sheet's or soft body's
        settle.
        """
        self.bridge = host_bridge(pipeline, self.name)
        self.join_settle(pipeline)

    # ---------------------------------------------------------------- runtime
    def positions(self, source=None) -> np.ndarray:
        """The centerline now, ``[n, 3]``, in order along the cable.

        Same name and shape as
        :meth:`~robosimstudio.domains.rod.entity.EntityRod.positions`. Pass the
        instant; the model's array is the as-built layout (see
        :meth:`Centerline.centerline`).

        Args:
            source: A :class:`newton.State`, or a ``particle_q`` array, or
                ``None`` for the model's own -- which is the as-built layout and
                is right only before anything has stepped.
        """
        if not len(self.particles):
            return np.zeros((0, 3), dtype=np.float64)
        if source is None:
            if self.model is None:
                return np.zeros((0, 3), dtype=np.float64)
            q = self.model.particle_q.numpy()
        elif hasattr(source, "particle_q"):
            q = source.particle_q.numpy()
        else:
            q = np.asarray(source)
        return np.asarray(q, dtype=np.float64)[self.particles]

    # ---------------------------------------------------------------- holding
    def hold(
        self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Pin the node nearest ``arm``'s jaws through the engine, riding the hand.

        The particle answer to :meth:`~robosimstudio.domains.rod.centerline.Centerline.hold`.
        Where the capsule chain's latch is a kernel pinning a *body* to a
        gripper frame, this asks the foreign engine for a compliant node
        constraint and has the engine carry it on the hand's body
        (``bridge.ride``): re-aimed before every engine substep from the hand's
        pose swept through the control step, as the mirrored fingers are, so the
        node keeps up with the hand rather than trailing it by a step.
        Returns the global node index taken, or ``-1``.
        """
        if world:
            # No per-world pin on the host engine yet.
            raise NotImplementedError(f"rod {self.name!r} on a host engine holds in world 0 only, not world {world}")
        if not on:
            entry = self._pins.pop(slot, None)
            if entry is not None and self.bridge is not None:
                self.bridge.unpin(entry[0])
            return -1
        if self.bridge is None:
            raise RuntimeError(f"rod {self.name!r} is not on a host engine; nothing to pin through")
        if state is None:
            raise ValueError("hold(True) needs the instant, to find the jaws")
        body_q = np.asarray(state.body_q.numpy() if hasattr(state.body_q, "numpy") else state.body_q)
        grip, hand_pos, _hand_quat = self.hand_tcp(arm, body_q)
        nodes = self.positions(state)
        local = self.nearest_to(nodes, hand_pos, at=at)
        if local < 0:
            return -1
        node = int(self.particles[local])
        from robosimstudio.core.math import quat_to_mat

        body = np.asarray(body_q[grip], dtype=np.float64)
        if slot in self._pins:
            self.hold(False, slot=slot)
        pin = self.bridge.pin_nodes(self.name, np.array([node]), nodes[local][None, :].copy())
        self.bridge.ride(pin, grip, (quat_to_mat(body[3:7]).T @ (nodes[local] - body[:3]))[None, :])
        self._pins[slot] = (int(pin), node)
        return node

    def holding(self, slot: int = 0) -> bool:
        """Whether ``slot`` holds a node, asked of the engine.

        A reset removes every engine pin, so it reads as released, as on the
        capsule chain. Nothing re-takes the node: the pin is an offset from
        where the hand was, and the reset sends the hand home. A caller that
        wants the cable in hand takes it again after the reset. A slot the
        engine let go of is forgotten here.
        """
        entry = self._pins.get(slot)
        if entry is None or self.bridge is None:
            return False
        if not self.bridge.has_pin(entry[0]):
            del self._pins[slot]
            return False
        return True

    def published_states(self, snapshot) -> dict[str, float]:
        """The same four keys the capsule chain publishes, so a goal reads either.

        The two latch flags always; ``n_segments`` and ``length_m`` from a
        snapshot this cable's nodes can be read off, and left out with none
        (a bare ``None``) rather than reported as a 0 m rod.
        """
        flags = {"held": float(self.holding(0)), "anchor_held": float(self.holding(1))}
        chain = self.centerline(snapshot) if snapshot is not None else np.zeros((0, 3))
        if len(chain) < 2:
            return flags
        length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
        return {**flags, "n_segments": float(len(chain) - 1), "length_m": length}

    def centerline(self, snapshot) -> np.ndarray:
        """This centerline's nodes, in order -- this implementation's :class:`Centerline`.

        From the probe (filled once per control step), not the model's
        as-built array. Indexed by this cable's particles, because
        ``ParticleProbe.positions`` holds every particle in the scene.
        """
        if not len(self.particles):
            return np.zeros((0, 3), dtype=np.float64)
        # A probe snapshot carries one probe per entity under ``particles``; a
        # ``HostSnapshot`` carries the whole model's ``particle_q`` and no
        # ``particles``.
        probes = getattr(snapshot, "particles", None)
        if probes is not None:
            probe = probes.get(self.name)
            if probe is None:
                return np.zeros((0, 3), dtype=np.float64)
            return np.asarray(probe.positions, dtype=np.float64)[self.particles]
        q = getattr(snapshot, "particle_q", None)
        if q is None or not len(q):
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(q, dtype=np.float64)[self.particles]

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's node positions. Once per control step."""
        if self.probe is None:
            raise RuntimeError(f"rod {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe

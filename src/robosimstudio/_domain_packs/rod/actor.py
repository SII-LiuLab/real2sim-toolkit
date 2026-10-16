# SPDX-License-Identifier: Apache-2.0
"""The rod as a foreign-engine actor: how one is read off the model, and how SuperDex builds it.

Two registrations, through seams the library keeps open for deformables it does
not ship:

* :func:`~robosimstudio.engines.geometry.register_actor_kind` -- how a ``"rod"``
  actor is **read** off the finalized model (:func:`read_rod`). Engine-neutral:
  a centerline is an ordered run of particles, and the order is the
  connectivity.
* :func:`~robosimstudio.engines.geometry.register_actor_builder` -- how
  **SuperDex** builds one (:func:`build_rod_actor`), with the parameter names
  that builder reads (:data:`CONSUMES`) so the composer can warn about one an
  entity states and nothing consumes.

A builder is handed the bridge and uses only what every builder gets:
``physics``, ``model``, ``scene_of``, ``actor_root``, ``contact_params`` and the
``actors`` built so far.

This module imports no engine: ``superdex`` is reached through the bridge, so
the ``rod`` domain imports on a checkout without the wheel.
"""

from __future__ import annotations

from itertools import combinations, pairwise
from typing import Any

import numpy as np

from robosimstudio.engines.geometry import ActorGeometry, actor_geometry, register_actor_builder, register_actor_kind

__all__ = ["CONSUMES", "anchor_dofs", "build_rod_actor", "read_rod", "teleport_rod", "transported_frames"]


def read_rod(model: Any, base: ActorGeometry) -> ActorGeometry:
    """A centerline: an ordered run of particles, and nothing to cut.

    Node order is the connectivity (the composer refuses a non-contiguous
    particle range). The engine builds the material frame from the nodes, or
    takes one the entity declared in ``params``; Newton stores no frame per
    particle.
    """
    del model
    if base.count < 2:
        raise ValueError(f"rod actor {base.name!r} has {base.count} node(s); a centerline needs at least 2")
    return base


def transported_frames(nodes: np.ndarray) -> np.ndarray:
    """One frame axis per element, parallel-transported along the polyline.

    The tubular skin is built around each element's frame axis, and the rest
    twist is read off consecutive axes. Transporting keeps the rest twist zero.
    A per-element axis rule ("perpendicular to the tangent and to z") flips 90
    degrees where a cable turns from vertical to horizontal, and the solver
    diverged on that twist from the first step (residual 5.7e3).
    """
    tangents = np.diff(np.asarray(nodes, dtype=np.float64), axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)
    first = tangents[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(first[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis = np.cross(first, helper)
    axis /= np.linalg.norm(axis)
    axes = [axis]
    for prev, cur in pairwise(tangents):
        # Rotate the axis by the rotation taking ``prev`` to ``cur`` (Rodrigues).
        k = np.cross(prev, cur)
        s, c = np.linalg.norm(k), float(np.dot(prev, cur))
        if s > 1.0e-12:
            k /= s
            axis = axis * c + np.cross(k, axis) * s + k * np.dot(k, axis) * (1.0 - c)
        axis = axis - np.dot(axis, cur) * cur
        axis /= np.linalg.norm(axis)
        axes.append(axis)
    return np.asarray(axes)


def anchor_dofs(
    nodes: np.ndarray, positions: np.ndarray, stride: int, twists: np.ndarray = ()
) -> tuple[np.ndarray, np.ndarray]:
    """DoF indices and world values that hold ``nodes`` where they are, sorted by DoF.

    A rod refuses node boundary conditions ("only supported by soft and shell
    actors") and takes them per DoF instead; its translation DoFs are the first
    three of each node's ``stride`` and their values are **world positions**,
    not displacements (measured: a value of 0 sent the node to the origin).

    ``twists`` names nodes whose fourth DoF is held too: the engine documents it
    as the twist angle of the element *after* that node, and takes only zero
    for it -- the element's material axes turning no more than its tangent
    makes them. Holding an end element's two nodes and its twist is a clamp.
    """
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    dofs = (nodes[:, None] * int(stride) + np.arange(3)[None, :]).reshape(-1)
    values = np.asarray(positions, dtype=np.float64)[nodes].reshape(-1)
    twists = np.asarray(twists, dtype=np.int64).reshape(-1)
    if twists.size and int(stride) >= 4:
        dofs = np.concatenate([dofs, twists * int(stride) + 3])
        values = np.concatenate([values, np.zeros(len(twists))])
    order = np.argsort(dofs, kind="stable")
    return dofs[order].astype(np.int32), values[order]


def build_rod_actor(
    bridge: Any, record: dict, particle_q: np.ndarray = None, tet_indices=None, tet_materials=None
) -> dict:
    """A cable as a SuperDex Cosserat rod.

    Consecutive nodes form one polyline element each
    (``create_mesh_shape(nodes_per_element=2, ...)``). Unlike a soft or shell
    actor:

    * **Four degrees of freedom per node**: a twist about the tangent beside
      the translation. The read-back slices the first three.
    * **Not readable by ``NodePositions``** (SuperDex supports it "for soft and
      shell actors" only). The configuration is the rest mesh plus a per-node
      displacement, so this actor carries its own reader.

    Stiffnesses are axial, flexural and torsional, plus a linear density.
    """
    physics = bridge.physics
    del particle_q, tet_indices, tet_materials  # the shared reader takes them off the model
    geom = actor_geometry(bridge.model, {**record, "kind": "rod"})
    start, count, params, rest = geom.start, geom.count, geom.params, geom.rest
    # ``rest`` is where the nodes start; the actor is built at its stress-free
    # shape, which is the same polyline unless the entity states another.
    free = np.asarray(params["stress_free"], dtype=np.float64) if "stress_free" in params else rest

    radius = float(params.get("radius", 0.005))
    if bool(params.get("skin", True)):
        # Contact on a tubular skin embedded on the polyline (as the engine's
        # tendon example builds a rod). Centerline samples alone let the axis
        # rest on the surface: +0.91 mm clearance for an 11 mm cable that
        # should show +11.0. The simulation mesh is still the polyline, so the
        # read-back is unchanged.
        axes = transported_frames(free)
        np_real = np.float64 if physics.uses_double_precision() else np.float32
        model = physics.experimental.generate_tubular_rod_model_data(
            nodes=free.astype(np_real),
            element_frame_axes=axes.astype(np_real),
            radius=radius,
            num_cross_section_segments=int(params.get("skin_segments", 8)),
            is_closed_loop=False,
        )
        shape = physics.create_model_shape(model)
    else:
        segments = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1).astype(np.int32)
        shape = physics.create_mesh_shape(
            nodes_per_element=2,
            coordinates=free.ravel().astype(np.float64),
            connectivity=segments.ravel(),
        )
    material = physics.experimental.RodMaterialParams()
    material.axial_stiffness = float(params.get("axial_stiffness", 1.0e5))
    flexural = params.get("flexural_stiffness", 2.0e-2)
    material.flexural_stiffness = (
        [float(v) for v in flexural] if isinstance(flexural, (list, tuple)) else [float(flexural)] * 2
    )
    material.torsional_stiffness = float(params.get("torsional_stiffness", 1.0e-2))
    # Linear density [kg/m]; EntityRodSuperdex.host_actor derives it from
    # density x cross-section.
    material.linear_density = float(params.get("linear_density", 0.19))
    # Rotational inertia per length [kg*m] -- rho * J for a circular section.
    # Left at the engine's zero, a node's twist has no mass behind it.
    inertia = params.get("linear_rotational_inertia")
    if inertia is None:
        inertia = 0.5 * material.linear_density * radius**2
    material.linear_rotational_inertia = float(inertia)
    material.mass_damping_coefficient = float(params.get("mass_damping", 0.0))
    material.stiffness_damping_coefficient = float(params.get("stiffness_damping", 1.0e-3))

    rod_params = physics.experimental.RodActorParams()
    rod_params.name = str(record.get("name", f"rod:{start}"))
    rod_params.shape = shape
    rod_params.material = material
    # Rest coordinates are already world, so at one world the root is the
    # identity. Replicated, the root carries this world's placement; see the
    # bridge's ``actor_root``.
    rod_params.world_from_local = bridge.actor_root(record)
    rod_params.has_gravity = True
    rod_params.layer = "soft"
    rod_params.collider_type = physics.ColliderType.POINT_CLOUD
    rod_params.point_cloud_collider.radius = radius
    rod_params.point_cloud_collider.self_contact = bool(params.get("self_contact", False))
    rod_params.use_visual_mesh_contact = bool(params.get("skin", True))
    samples = params.get("skin_samples")
    if samples:
        # Quadrature points per skin triangle. A skin of 8 sides on 5 mm
        # elements samples every ~2.4 mm round and 5 mm along: a polymer
        # clip's 1.3 mm lip slipped between those samples and sat 1.4 mm
        # inside the cable (measured), where every sample was outside it.
        rod_params.visual_mesh_contact_element_type = getattr(physics.ActorBoundaryElementType, str(samples))
    rod_params.contact = bridge.contact_params(params, penalty=8.0e7, friction=0.9, padding=2.0e-3)
    actor = physics.experimental.create_rod_actor(bridge.scene_of(record.get("world", 0)), rod_params)

    reference = np.asarray(actor.get_mesh().coordinates, dtype=np.float64).reshape(-1, 3)
    stride = max(1, len(np.asarray(actor.get_displacements())) // max(1, len(reference)))
    if free is not rest:
        teleport_rod(bridge, record, actor, rest, stride)

    anchors = np.asarray(params.get("anchors", ()), dtype=np.int64)
    twists = np.asarray(params.get("anchor_twists", ()), dtype=np.int64)
    if anchors.size or twists.size:
        dofs, values = anchor_dofs(anchors, rest, stride, twists)
        actor.add_boundary_condition_dofs_world_permanent(dof_indices=dofs, dof_values_world=values)

    def read_nodes() -> np.ndarray:
        """Rest plus the translation half of each node's degrees of freedom."""
        return reference + np.asarray(actor.get_displacements()).reshape(-1, stride)[:, :3]

    np_real = np.float64 if physics.uses_double_precision() else np.float32
    return {
        "name": str(record.get("name", "")),
        "actor": actor,
        "read": read_nodes,
        "start": start,
        "count": count,
        "prev": rest.astype(np_real).copy(),
    }


def teleport_rod(bridge: Any, record: dict, actor: Any, rest: np.ndarray, stride: int) -> None:
    """Carry a just-built rod from its stress-free shape onto its starting polyline, and move nothing else.

    A rod takes no displacements or node positions (both are "soft and shell"
    only), so every translation DoF is held at its start for two 1 us steps (the
    second zeroes the speed the first implied), then released. Exact to the
    float (0.000 mm on a 230-node cable); the twist settles on the first real
    step. Runs before the welds, which join the rod where it now is.

    The steps are whole-scene steps, so every actor already built in this world
    is isolated for them:

    * Contact is off between every pair of actors in the world and on the rod's
      self pair. Any live contact aborts the process at the next step ("Expected
      a force vector for every contact between this pair of entities", SIGILL,
      exit 132); seen with a cable joined to a weight, a cable pressed into a
      soft block, and the rod's own self contact (an L, a U, a three-turn
      helix). Contact is on by default and nothing turns it off before the
      bridge's ``_build_welds``, so every pair is switched back on. The self
      pair goes back on for this rod only, and its collider's ``self_contact``
      still decides whether it touches itself (an 8 mm-pitch helix moves 2.04 mm
      in ten steps with it, 12.40 mm without). Other self pairs are left alone:
      an earlier rod in self contact (4 mm-pitch helix) does not trip the abort,
      and re-enabling a skinned body's self pair would enable its adjacent
      links, which the engine disables at creation.
    * Gravity is zero.
    * Every earlier rod is held at its current translation DoFs, except those
      its anchors already hold. A stress-free rod still carries elastic stress:
      unheld, a second teleport left the first 3.2 um off and moving at
      1.83 m/s (E = 1e9).
    * Afterwards every actor is set to rest, which also resets the BDF2 history
      (it falls back to backward Euler). A body on a rigid core has its soft
      part zeroed whole and its skeleton's joint velocities set to zero.

    What to hold and reset is read off the engine (type, mesh, displacements,
    boundary conditions), not off builder entries, so an actor kind another
    builder added is isolated as far as its type allows.

    Diffing the engine's whole-scene state across each rod's build, every
    earlier actor (a joined weight and cable, a cable in a block, a body on a
    rigid core, three stress-free rods) keeps bit-identical displacements,
    velocities, rod frames and integrator history; only solver caches differ.

    Contact, gravity and the holds are restored in a ``finally``.
    """
    physics = bridge.physics
    world = int(record.get("world", 0))
    scene = bridge.scene_of(world)
    earlier = [e for e in bridge.actors if int(e["world"]) == world]
    handles = [actor.get_handle(), *(e.get("skinned", e["actor"]).get_handle() for e in earlier)]
    pairs = list(combinations(handles, 2))
    dofs, values = anchor_dofs(np.arange(len(rest)), rest, stride)
    holds = [(actor, dofs, values)]
    for entry in earlier:
        other = entry["actor"]
        if "skinned" in entry or other.get_type() != physics.ActorType.ROD:
            continue
        reference = np.asarray(other.get_mesh().coordinates, dtype=np.float64).reshape(-1, 3)
        displacements = np.asarray(other.get_displacements(), dtype=np.float64).reshape(len(reference), -1)
        nodes = reference + displacements[:, :3]
        held_dofs, held_values = anchor_dofs(np.arange(len(nodes)), nodes, displacements.shape[1])
        # A DoF entered twice takes the later value; its anchors already hold their own.
        free = ~np.isin(held_dofs, np.asarray(other.get_boundary_condition_dof_indices()))
        holds.append((other, held_dofs[free], held_values[free]))
    gravity = [float(v) for v in scene.get_gravity()]
    own = actor.get_handle()
    try:
        for a, b in pairs:
            scene.enable_actor_contact_symmetric(a, b, False, physics.IncludeNestedActors.YES)
        scene.enable_actor_contact_symmetric(own, own, False, physics.IncludeNestedActors.NO)
        scene.set_gravity([0.0, 0.0, 0.0])
        for _ in range(2):
            for held, held_dofs, held_values in holds:
                held.clear_boundary_conditions()
                held.add_boundary_condition_dofs_world(dof_indices=held_dofs, dof_values_world=held_values)
            scene.step(1.0e-6)
    finally:
        scene.set_gravity(gravity)
        for a, b in pairs:
            scene.enable_actor_contact_symmetric(a, b, True, physics.IncludeNestedActors.YES)
        scene.enable_actor_contact_symmetric(own, own, True, physics.IncludeNestedActors.NO)
        for held, _, _ in holds:
            held.clear_boundary_conditions()
    nodal = (physics.ActorType.SOFT, physics.ActorType.SHELL, physics.ActorType.ROD)
    for entry in [{"actor": actor}, *earlier]:
        target = entry["actor"]
        if "skinned" in entry:
            target.set_zero_displacements_and_velocities()
            skeleton = entry["skinned"]
            skeleton.set_articulated_joint_velocities(np.zeros(skeleton.get_num_dofs()))
        elif target.get_type() in nodal:
            target.set_node_velocities_local(np.zeros(len(np.asarray(target.get_displacements()))))
        # Any other type takes no node velocity; contact and gravity off, nothing moved it.


CONSUMES = (
    "anchor_twists",
    "anchors",
    "axial_stiffness",
    "contact_padding",
    "contact_penalty",
    "contact_smoothing",
    "contact_threshold",
    "flexural_stiffness",
    "friction",
    "friction_falloff",
    "linear_density",
    "linear_rotational_inertia",
    "mass_damping",
    "radius",
    "self_contact",
    "skin",
    "skin_samples",
    "skin_segments",
    "stiffness_damping",
    "stress_free",
    "torsional_stiffness",
    "welds",
)
"""What :func:`build_rod_actor` reads off ``HostActor.params``; the composer warns
on anything else. ``welds`` is read by the bridge's weld pass, not the builder."""

register_actor_kind("rod", read_rod, summary="A centerline: ordered nodes, the order being the connectivity.")
register_actor_builder("superdex", "rod", build_rod_actor, consumes=CONSUMES, summary="SuperDex builds a rod actor.")

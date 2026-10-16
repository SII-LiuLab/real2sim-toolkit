# SPDX-License-Identifier: Apache-2.0
"""The ``rod`` kind -- the rope: what it is, what builds it, what reads it.

Importing this registers the rod domain, its entities and measures. The
``DOMAINS`` package scan imports it, so the domain exists before any goal binds.

This directory is a pack: nothing in ``robosimstudio/`` names it, and a tree may
leave it out. Each module reaches the library by registering:

* ``entity`` / ``superdex`` -- the two entities, by ``register_entity("rod", engine=...)``;
* ``centerline`` -- the ``Centerline`` capability, by ``register_capability``;
* ``actor`` -- how a ``"rod"`` actor is read off the model and how SuperDex builds
  one, by ``register_actor_kind`` and ``register_actor_builder``;
* ``features`` -- the arc-length ``point`` feature kind, by ``register_kind``;
* ``measures`` -- ``winding_turns``, ``straightness`` and the layout's own
  ``height`` and ``speed``, by ``register_atom`` and ``handler``;
* ``materials`` -- the rod material kind and its preset;
* ``latch`` -- the grasp latch's kernel and device state. No registry: the entity
  uses it, and a repro bundle copies it.

The ``"rod"`` state layout is minted by ``register_domain(reads="rod")`` below.
The pack takes the post from the library: :class:`EntityPeg` is the static
domain's ``EntityPost`` under the cable suites' name.
``tests/test_rod_is_a_pack.py`` checks all of this in both directions.
"""

from __future__ import annotations

import sys as _sys

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import register_domain

from . import actor as _actor  # noqa: F401 -- registers how a rod actor is read, and how SuperDex builds one
from . import latch as _latch
from . import materials as _materials  # noqa: F401 -- registers the rod material kind and its preset
from . import measures as _measures  # noqa: F401 -- registers winding_turns and straightness
from .centerline import Centerline
from .declare import Rod
from .entity import (
    CableDriveSpec,
    CableLatchSpec,
    EntityCable,
    EntityPeg,
    EntityPegRound,
    EntityRod,
    SpecCable,
    fixture_shape_cfg,
)
from .features import KIND_NAMES, RodParam  # registers the arc-length ``point`` feature kind
from .make import rod
from .spec import RodMaterialSpec, RodSpec

__all__ = [
    "KIND_NAMES",
    "CableDriveSpec",
    "CableLatchSpec",
    "Centerline",
    "EntityCable",
    "EntityPeg",
    "EntityPegRound",
    "EntityRod",
    "EntityRodSuperdex",
    "Rod",
    "RodMaterialSpec",
    "RodParam",
    "RodSpec",
    "SpecCable",
    "fixture_shape_cfg",
    "rod",
]

# Old path, still named as a string by cable_route's
# ReproHooksSpec(module="robosimstudio.solvers.grasp.latch"); the bundle exporter
# imports it to find the file it copies. setdefault, so a real module of that
# name wins. Drop when the suite names robosimstudio.domains.rod.latch.
_sys.modules.setdefault("robosimstudio.solvers.grasp.latch", _latch)

DOMAIN = register_domain(
    "rod",
    # Both: Newton's capsule chain owns bodies, joints and shapes; SuperDex's
    # Cosserat centerline owns particles. The composer checks a backend against
    # what each entity owns, so a capsule chain on a particles-only solver still
    # fails and names the state it could not own.
    state=("bodies", "joints", "particles", "shapes"),
    reads="rod",
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A cable: a capsule chain on Newton, a Cosserat centerline on SuperDex.",
)

# The library registers only the generic "rod", so a spec can ask for one
# without a task having bound it. Task names (cable_rod, hang_line, hang_post)
# stay in their tasks, as separate registry keys onto the same classes. "post"
# is the static domain's, so rss.post builds without this pack.
register_entity("rod")(EntityRod)

# The SuperDex implementation, selected by ("rod", backend("superdex")); the
# spec is the same either way.
from .superdex import EntityRodSuperdex

register_entity("rod", engine="superdex")(EntityRodSuperdex)

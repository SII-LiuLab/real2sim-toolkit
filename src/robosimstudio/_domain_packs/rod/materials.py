# SPDX-License-Identifier: Apache-2.0
"""A rod's matter, in the registry that holds materials.

Registers the rod material kind
(:class:`~robosimstudio.domains.rod.spec.RodMaterialSpec`) and one record: the
reference cable's drive gains as ``RodSpec`` defaults them. The record is a
read-back of the built model, so a rod has no ``material`` field (see
``RodMaterialSpec``).
"""

from __future__ import annotations

from robosimstudio.assetlib.materials import MaterialRecord, register_material, register_material_kind

from .spec import RodMaterialSpec

__all__ = ["REFERENCE_CABLE_DRIVE"]

register_material_kind("rod", RodMaterialSpec)

REFERENCE_CABLE_DRIVE = register_material(
    MaterialRecord(
        name="reference_cable_drive",
        spec=RodMaterialSpec(),
        notes="The shipped capsule-chain cable's drive gains, as RodSpec defaults them; a starting point, not a measurement.",
    )
)

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from robocasa_real2sim.catalog import scan_local_assets, select_assets
from robocasa_real2sim.isaac_prepare import prepare_mjcf
from robocasa_real2sim.validation import load_mjcf
from robocasa_real2sim.usd_backend import build_mesh_usd, inspect_usd, prepare_usd


def test_usd_index_requires_declared_topology_and_filters_behavior(tmp_path):
    metadata = {"id": "machine", "category": "conveyor", "model": "asset.usda", "license": "local", "behavior": "static"}
    (tmp_path / "asset.usda").write_text("#usda 1.0\n")
    (tmp_path / "asset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="declared joints"):
        scan_local_assets(tmp_path)
    metadata["joints"] = []
    (tmp_path / "asset.json").write_text(json.dumps(metadata))
    candidates = scan_local_assets(tmp_path)
    result = select_assets({"id": "x", "category": "conveyor", "behavior": "conveyor"}, candidates)
    assert not result["candidates"]
    assert "behavior_mismatch" in result["rejected"][0]["reasons"]


def test_prepared_mjcf_preserves_nontrivial_com_and_inertia_after_move(tmp_path):
    pytest.importorskip("mujoco")
    source = tmp_path / "input.xml"
    source.write_text('''<mujoco><compiler angle="degree" inertiagrouprange="0 0"/>
      <worldbody><body><joint name="door" range="0 90"/>
      <geom type="box" size=".2 .01 .4" pos=".2 0 .4" density="650"/>
      <geom type="box" size="1 1 1" group="1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    original = load_mjcf(source)
    output = tmp_path / "prepared"
    report = prepare_mjcf(source, output)
    moved = tmp_path / "portable"
    shutil.move(str(output), str(moved))
    frozen = load_mjcf(moved / "model.xml")
    for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "jnt_range"):
        np.testing.assert_allclose(getattr(original, field), getattr(frozen, field))
    assert report["inertia_preserved_in_mujoco"] and not report["isaac_runtime_validated"]
    assert frozen.body_mass[1] < 10


@pytest.fixture
def mesh_request(tmp_path):
    pytest.importorskip("pxr")
    trimesh = pytest.importorskip("trimesh")
    mesh = tmp_path / "object.ply"
    trimesh.creation.box(extents=[100, 200, 300]).export(mesh)
    return {"id": "actual_input", "category": "household_object", "geometry": {"parts": [
        {"name": "Object", "mesh": str(mesh), "scale_m_per_unit": .001, "aligned_z_up": True,
         "physics": "rigid", "mass_kg": .5, "appearance": {"color_rgb": [.4, .5, .6]}}
    ]}}


def test_mesh_export_preserves_metric_bounds_mass_and_portability(tmp_path, mesh_request):
    from pxr import Usd, UsdGeom, UsdPhysics
    output = tmp_path / "output"
    report = build_mesh_usd(mesh_request, output)
    Path(mesh_request["geometry"]["parts"][0]["mesh"]).rename(tmp_path / "hidden.ply")
    moved = tmp_path / "moved"
    shutil.move(str(output), str(moved))
    stage = Usd.Stage.Open(str(moved / "asset.usda"))
    extent = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Object/Visual")).GetExtentAttr().Get()
    np.testing.assert_allclose(np.array(extent[1]) - extent[0], [.1, .2, .3])
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath("/Asset/Object")).GetMassAttr().Get() == .5
    structure = inspect_usd(moved / "asset.usda")
    assert structure["meters_per_unit"] == 1 and structure["up_axis"] == "Z"
    assert structure["rigid_bodies"] == ["/Asset/Object"] and len(structure["colliders"]) == 1
    assert report["status"] == "prepared" and not report["isaac_runtime_validated"]


@pytest.mark.parametrize("change,match", [({"scale_m_per_unit": 0}, "scale_m_per_unit"),
                                          ({"aligned_z_up": False}, "Align"),
                                          ({"requires_open_interior": True}, "closes the interior")])
def test_unknown_scale_frame_or_closed_receptacle_cannot_export(tmp_path, mesh_request, change, match):
    mesh_request["geometry"]["parts"][0].update(change)
    with pytest.raises(ValueError, match=match):
        build_mesh_usd(mesh_request, tmp_path / "output")
    report = json.loads((tmp_path / "output/manifest.json").read_text())
    assert report["status"] == "failed" and not report["isaac_runtime_validated"]


def test_usd_conveyor_velocity_uses_source_units_and_survives_move(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    source = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    UsdGeom.SetStageUpAxis(stage, "Z")
    belt = UsdGeom.Cube.Define(stage, "/Belt").GetPrim()
    UsdPhysics.CollisionAPI.Apply(belt)
    stage.GetRootLayer().Save()
    report = prepare_usd(source, tmp_path / "prepared", "/Belt", .25)
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "prepared"), str(moved))
    result = Usd.Stage.Open(str(moved / "asset.usda"))
    prim = result.GetPrimAtPath("/Belt")
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
    np.testing.assert_allclose(prim.GetAttribute("physxSurfaceVelocity:surfaceVelocity").Get(), [25, 0, 0])
    assert inspect_usd(moved / "asset.usda")["belts"] == ["/Belt"]
    assert not report["isaac_runtime_validated"]


def test_usd_packaging_keeps_texture_dependency_after_originals_move(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    source_folder = tmp_path / "source"
    source_folder.mkdir()
    (source_folder / "surface.png").write_bytes(b"test dependency bytes")
    source = source_folder / "input.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    shader = UsdShade.Shader.Define(stage, "/Texture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("surface.png"))
    stage.GetRootLayer().Save()
    prepare_usd(source, tmp_path / "prepared")
    source_folder.rename(tmp_path / "source_hidden")
    result = Usd.Stage.Open(str(tmp_path / "prepared/asset.usda"))
    value = result.GetPrimAtPath("/Texture").GetAttribute("inputs:file").Get()
    assert value.path.startswith("assets/") and Path(value.resolvedPath).read_bytes() == b"test dependency bytes"


def test_usd_without_authored_scale_is_rejected(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd
    source = tmp_path / "ambiguous.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/Object", "Xform")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="explicitly declare"):
        prepare_usd(source, tmp_path / "output")


def test_missing_usd_composition_dependency_cannot_be_packaged(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    source = tmp_path / "missing.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.DefinePrim("/Asset").GetReferences().AddReference("nonexistent.usda")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unresolved composition"):
        prepare_usd(source, tmp_path / "output")


def test_nested_belt_rigid_body_is_rejected(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    source = tmp_path / "nested.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/Machine", "Xform"))
    belt = UsdGeom.Cube.Define(stage, "/Machine/Belt").GetPrim()
    UsdPhysics.CollisionAPI.Apply(belt)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="nested under"):
        prepare_usd(source, tmp_path / "output", "/Machine/Belt", .2)

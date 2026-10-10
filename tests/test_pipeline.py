from pathlib import Path
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from robocasa_real2sim.capture import inspect_captures
from robocasa_real2sim.catalog import native_catalog, select_assets, scan_local_assets, mjcf_joint_types
from robocasa_real2sim.reconstruction import reconstruction_plan, command_plan, stage_inputs, check_dense_geometry
from robocasa_real2sim.appearance import rank_textures
from robocasa_real2sim.adapters import build_native
from robocasa_real2sim.validation import load_mjcf, validate_asset


def test_joint_topology_takes_priority_over_appearance():
    request = {"id": "cab", "category": "cabinet", "joint_types": ["hinge", "hinge"], "geometry": {"panel_type": "slab"}}
    result = select_assets(request, native_catalog())
    assert [item["id"] for item in result["candidates"]] == ["robocasa:hinge_cabinet:slab"]
    assert any("joint_topology_mismatch" in item["reasons"] for item in result["rejected"])
    assert result["candidates"][0]["appearance_score"] is None


def test_state_isolation_and_duplicate_frames(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (80, 80), "tan").save(image)
    captures = [{"path": str(image), "state": state} for state in ("closed", "closed", "open")]
    report = inspect_captures({"id": "cab", "captures": captures})
    assert [item["accepted"] for item in report["images"]] == [True, False, True]
    assert reconstruction_plan(report, "closed")["status"] == "capture_required"
    assert reconstruction_plan(report, "open")["status"] == "capture_required"


def test_mask_names_and_paths_are_not_shell_strings(tmp_path):
    commands = command_plan("colmap.exe", tmp_path / "path with spaces", True)
    assert "--ImageReader.mask_path" in commands[0]
    assert commands[0][commands[0].index("--image_path") + 1].endswith("path with spaces/images") or commands[0][commands[0].index("--image_path") + 1].endswith("path with spaces\\images")


def test_urdf_fixed_links_are_not_counted_as_movable_parts(tmp_path):
    (tmp_path / "asset.json").write_text(json.dumps({"id": "test", "category": "cabinet", "model": "mobility.urdf", "license": "test-only"}))
    (tmp_path / "mobility.urdf").write_text('<robot name="cab"><joint name="shelf" type="fixed"/><joint name="door" type="revolute"/></robot>')
    assert scan_local_assets(tmp_path)[0]["joints"] == ["hinge"]


def test_mjcf_defaults_describe_types_but_are_not_additional_joints():
    from xml.etree.ElementTree import fromstring
    root = fromstring('''<mujoco><default><joint damping="1"/>
      <default class="drawer"><joint type="slide"/></default></default>
      <worldbody><body childclass="drawer"><joint name="drawer"/>
        <body><joint name="door" class="main"/></body></body></worldbody></mujoco>''')
    assert mjcf_joint_types(root) == ["slide", "hinge"]


def test_resume_rejects_changed_masks_or_photo_count_without_overwriting(tmp_path):
    image, mask = tmp_path / "photo.png", tmp_path / "mask.png"
    Image.new("RGB", (80, 80), "tan").save(image)
    Image.new("L", (80, 80), 255).save(mask)
    work = tmp_path / "job"
    for folder in ("images", "masks"):
        (work / folder).mkdir(parents=True)
    plan = {"images": [{"path": str(image), "mask": str(mask)}]}
    stage_inputs(plan, work)
    staged_mask = (work / "masks/00000.png.png").read_bytes()
    previous = {"masked": True}
    stage_inputs(plan, work, previous)
    Image.new("L", (80, 80), 0).save(mask)
    with pytest.raises(ValueError, match="masks changed"):
        stage_inputs(plan, work, previous)
    assert (work / "masks/00000.png.png").read_bytes() == staged_mask
    with pytest.raises(ValueError, match="photo count changed"):
        stage_inputs({"images": plan["images"] * 2}, work, previous)


def test_texture_screening_distinguishes_color_and_grain_direction(tmp_path):
    vertical = np.full((96, 96, 3), [198, 183, 155], dtype=np.uint8)
    vertical[:, ::3] -= 20
    horizontal = vertical.transpose(1, 0, 2)
    paths = [tmp_path / name for name in ("vertical.png", "horizontal.png", "dark.png")]
    for path, data in zip(paths, (vertical, horizontal, vertical // 3)):
        Image.fromarray(data).save(path)
    request = {"appearance": {"reference": {"path": str(paths[0])}}}
    report = rank_textures(request, paths)
    assert [Path(item["texture"]).name for item in report["candidates"]] == ["vertical.png", "horizontal.png", "dark.png"]
    assert not report["appearance_calibrated"]
    request["appearance"]["reference"]["bbox"] = [0.8, 0, 0.2, 1]
    with pytest.raises(ValueError, match="bbox"):
        rank_textures(request, paths)


@pytest.mark.parametrize("candidate", native_catalog(), ids=lambda item: item["id"])
def test_export_remains_portable_and_doors_open_without_self_collision(tmp_path, candidate):
    pytest.importorskip("robocasa")
    pytest.importorskip("mujoco")
    request = {"id": "cab", "category": candidate["category"], "joint_types": candidate["joints"],
               "geometry": {"size_m": [1.0, 0.45, 0.3 if candidate["category"] == "drawer" else 1.55],
                            "panel_type": candidate["panel_type"]}}
    source = tmp_path / "export"
    manifest = build_native(request, candidate, source)
    moved = tmp_path / "another directory" / "export"
    shutil.copytree(source, moved)
    model = load_mjcf(moved / "model.xml")
    assert model.njnt == len(candidate["joints"])
    assert np.max(np.abs(model.jnt_range)) > (0.4 if candidate["category"] == "drawer" else 1.5)
    if candidate["category"] == "drawer":
        # The default-density transparent interior region would add ~85 kg.
        assert model.body_mass[model.jnt_bodyid[0]] < 2
    validation = validate_asset(moved / "model.xml")
    assert validation["poses_checked"] == 19 ** len(candidate["joints"])
    assert validation["max_self_penetration_m"] < 1e-5
    assert not manifest["real_world_calibrated"]


def test_failed_validation_replaces_a_previous_success_report(tmp_path):
    pytest.importorskip("mujoco")
    (tmp_path / "validation.json").write_text('{"passed": true}')
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><worldbody><geom type="box" size="0 0 0"/></worldbody></mujoco>')
    with pytest.raises(ValueError):
        validate_asset(model)
    assert json.loads((tmp_path / "validation.json").read_text())["passed"] is False


def test_empty_mesh_cannot_pass_despite_a_nonempty_point_cloud(tmp_path):
    cloud, mesh = tmp_path / "cloud.ply", tmp_path / "surface.ply"
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n")
    mesh.write_text("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n")
    with pytest.raises(RuntimeError, match="empty surface"):
        check_dense_geometry(cloud, mesh)


def test_pipeline_failure_replaces_a_previous_success_report(tmp_path):
    pytest.importorskip("robocasa")
    from robocasa_real2sim.cli import main
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"id": "invalid", "category": "cabinet", "joint_types": ["hinge", "hinge"],
                                   "geometry": {"panel_type": "slab", "size_m": [1, -1, 1]}}))
    output = tmp_path / "output"
    output.mkdir()
    (output / "pipeline.json").write_text('{"status": "preview_asset_built"}')
    assert main(["pipeline", "--request", str(request), "--output", str(output)]) == 2
    assert json.loads((output / "pipeline.json").read_text())["status"] == "failed"

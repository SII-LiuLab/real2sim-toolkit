"""Build native fixtures and export portable MJCF with asset provenance."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .io import write_json


def adjust_door_pivots(root, width, gap):
    for joint in root.iter("joint"):
        if "doorhinge" not in joint.get("name", ""):
            continue
        lo, hi = [float(x) for x in joint.get("range").split()]
        pos = [float(x) for x in joint.get("pos", "0 0 0").split()]
        pos[0] = -(width - gap) / 2 if lo < 0 else (width - gap) / 2
        joint.set("pos", " ".join(map(str, pos)))


def bundle_assets(root, output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sources = []
    for asset in root.findall("asset/*"):
        if not asset.get("file"):
            continue
        source = Path(asset.get("file"))
        if not source.is_file():
            raise FileNotFoundError(f"Missing asset: {source}")
        digest = sha256(source.read_bytes()).hexdigest()
        destination = assets / f"{digest[:16]}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        asset.set("file", destination.relative_to(output).as_posix())
        sources.append({"path": asset.get("file"), "source": str(source), "sha256": digest})
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    return sources


def build_native(request, candidate, output):
    from robocasa.models.fixtures.cabinets import HingeCabinet, SingleCabinet, Drawer

    classes = {"hinge_cabinet": HingeCabinet, "single_cabinet": SingleCabinet, "drawer": Drawer}
    geometry = deepcopy(request.get("geometry", {}))
    size = geometry.get("size_m")
    if not isinstance(size, list) or len(size) != 3 or any(not np.isfinite(x) or x <= 0 for x in size):
        raise ValueError("geometry.size_m must contain three finite positive values")
    thickness = geometry.get("thickness_m", 0.02)
    gap = geometry.get("door_gap_m", 0.003)
    if not (0 < thickness < min(size) / 2 and 0 < gap < min(size) / 2):
        raise ValueError("Panel thickness and gap must fit inside the object dimensions")
    kind = candidate["fixture_type"]
    cfg = dict(name="cabinet_photo", size=size, thickness=thickness, door_gap=gap,
               panel_type=geometry.get("panel_type", candidate["panel_type"]),
               texture=geometry.get("texture", "textures/wood/wood_grain_1.png").replace("\\", "/"),
               handle_type=geometry.get("handle_type", "boxed"),
               handle_config={"texture": "textures/flat/black.png", "length": geometry.get("handle_length_m", min(0.35, size[2] * 0.7))},
               panel_config={"handle_vpos": "center", "handle_hpercent": 0.92})
    if kind != "drawer":
        cfg["num_levels"] = geometry.get("levels", 3)
    if cfg["panel_type"] in ("shaker", "raised"):
        # Native trim_th defaults to 20 mm, which consumes a 20 mm door entirely.
        # Keep both the trim and backing panel at positive thickness.
        panel_width = size[0] / (2 if kind == "hinge_cabinet" else 1) - gap
        cfg["panel_config"].update(trim_th=thickness / 2,
                                   trim_size=min(0.08, min(panel_width, size[2] - gap) * 0.15))
    if kind == "single_cabinet":
        cfg["orientation"] = geometry.get("orientation", "right")
    fixture = classes[kind](**cfg)
    root = ET.fromstring(ET.tostring(fixture.root))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    # Match robosuite/models/assets/base.xml: visual and region geoms do not
    # contribute mass/inertia. A drawer's invisible interior region is group 1.
    compiler.set("inertiagrouprange", "0 0")
    compiler.set("autolimits", "true")
    if kind != "drawer":
        adjust_door_pivots(root, size[0], gap)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = bundle_assets(root, output)
    ET.indent(root)
    ET.ElementTree(root).write(output / "model.xml", encoding="utf-8", xml_declaration=True)
    # Scene coordinates are separate from the local asset-centre coordinate frame.
    layout_cfg = {key: value for key, value in cfg.items() if key != "name"}
    layout_cfg.update(name="cabinet", type=kind, pos=[0.0, 0.175, size[2] / 2 + 0.075])
    (output / "fixture.yaml").write_text(yaml.safe_dump({"photo": {"cabinets": [layout_cfg]}}, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1, "object_id": request["id"], "route": "reuse",
        "candidate": candidate["id"], "units": "m", "frame": "x right, y back, z up; origin at fixture centre",
        "model": "model.xml", "assets": sources, "size_m": size,
        "dimension_source": geometry.get("dimension_source", "assumed"),
        "real_world_calibrated": geometry.get("dimension_source") == "measured",
        "appearance_source": "existing RoboCasa texture", "appearance_calibrated": False,
        "selected_texture": cfg["texture"],
        "appearance_selection": request.get("appearance", {}).get("selection_source", "configured or default texture"),
        "physics_source": "native fixture defaults; mass and friction are not measured",
        "compiler_settings": dict(compiler.attrib),
        "review_required": True, "joints": candidate["joints"],
        "captures": request.get("captures", []),
    }
    # Even measured dimensions do not establish calibrated mass or dynamics.
    manifest["real_world_calibrated"] = False
    write_json(output / "manifest.json", manifest)
    return manifest

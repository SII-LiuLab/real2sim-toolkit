"""Freeze MuJoCo-computed body inertia before importing an expanded MJCF."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .adapters import bundle_assets
from .io import write_json
from .validation import load_mjcf


def prepare_mjcf(source, output):
    import mujoco
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.parent == output:
        raise ValueError("Use a separate output folder to preserve the source MJCF")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "manifest.json"
    write_json(report_path, {"status": "running", "isaac_runtime_validated": False})
    try:
        root = ET.parse(source).getroot()
        if root.tag != "mujoco" or any(True for _ in root.iter("include")):
            raise ValueError("Provide expanded MJCF (includes must be resolved first)")
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        base = Path(compiler.get("assetdir", ""))
        for asset in root.findall("asset/*"):
            if asset.get("file"):
                directory = compiler.get("texturedir" if asset.tag == "texture" else "meshdir", str(base))
                path = Path(asset.get("file"))
                if not path.is_absolute():
                    path = source.parent / directory / path
                asset.set("file", str(path.resolve()))
        assets = bundle_assets(root, output)
        used = {node.get("name") for node in root.iter() if node.get("name")}
        for tag in ("body", "joint", "freejoint"):
            nodes = root.findall(f"worldbody/.//{tag}")
            for i, node in enumerate(nodes):
                if node.get("name"):
                    continue
                name = f"r2s_{tag}_{i}"
                while name in used:
                    name += "_"
                node.set("name", name)
                used.add(name)
        model_path = output / "model.xml"
        ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
        model = load_mjcf(model_path)
        bodies = []
        for node in root.findall("worldbody/.//body"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node.get("name"))
            mass = float(model.body_mass[body_id])
            if mass <= 0:
                continue
            existing = node.find("inertial")
            if existing is not None:
                node.remove(existing)
            # Quaternions are unitless, diagonal inertia is kg*m^2, positions are m.
            ET.SubElement(node, "inertial", mass=f"{mass:.17g}",
                          pos=" ".join(f"{x:.17g}" for x in model.body_ipos[body_id]),
                          quat=" ".join(f"{x:.17g}" for x in model.body_iquat[body_id]),
                          diaginertia=" ".join(f"{x:.17g}" for x in model.body_inertia[body_id]))
            bodies.append({"name": node.get("name"), "mass_kg": mass})
        # Explicit inertials must take precedence over geom inference.
        compiler.set("inertiafromgeom", "false")
        ET.indent(root)
        ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
        frozen = load_mjcf(model_path)
        for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
            if not np.allclose(getattr(model, field), getattr(frozen, field), rtol=1e-9, atol=1e-10):
                raise RuntimeError(f"Explicit inertia changed compiled {field}")
        types = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
                 int(mujoco.mjtJoint.mjJNT_BALL): "ball", int(mujoco.mjtJoint.mjJNT_FREE): "free"}
        joints = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                   "type": types[int(model.jnt_type[i])], "limited": bool(model.jnt_limited[i]),
                   "range_radians_or_m": model.jnt_range[i].tolist()} for i in range(model.njnt)]
        manifest = {"status": "prepared", "model": "model.xml", "bodies": bodies, "joints": joints,
                    "assets": [{"path": item["path"], "sha256": item["sha256"]} for item in assets],
                    "inertia_preserved_in_mujoco": True, "isaac_runtime_validated": False,
                    "real_world_calibrated": False, "physics_source": "MuJoCo compilation of source MJCF"}
        write_json(report_path, manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        write_json(report_path, {"status": "failed", "error": str(error), "isaac_runtime_validated": False})
        raise

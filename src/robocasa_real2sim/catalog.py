"""Explainable structural screening before appearance ranking."""

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from .io import read_json


def native_catalog():
    # Parametric templates are evaluated for structure. They are not photo matches.
    rows = []
    for category, kind, count in (("cabinet", "hinge_cabinet", 2),
                                  ("cabinet", "single_cabinet", 1),
                                  ("drawer", "drawer", 1)):
        for panel in ("slab", "shaker", "raised"):
            rows.append({
                "id": f"robocasa:{kind}:{panel}", "source": "robocasa",
                "category": category, "fixture_type": kind, "panel_type": panel,
                "joints": ["slide" if kind == "drawer" else "hinge"] * count,
                "behavior": "articulated",
                "parametric": True, "dimension_fit": "adaptable",
                "appearance_score": None, "appearance_evaluated": False,
            })
    return rows


def mjcf_joint_types(root):
    """Count body joints, resolving types inherited from MJCF default classes."""
    defaults = {"main": "hinge"}
    def read_default(node, inherited):
        joint = node.find("joint")
        kind = joint.get("type", inherited) if joint is not None else inherited
        defaults[node.get("class", "main")] = kind
        for child in node.findall("default"):
            read_default(child, kind)
    for node in root.findall("default"):
        read_default(node, "hinge")
    joints = []
    def read_body(node, inherited):
        current = node.get("childclass", inherited)
        for joint in node.findall("joint"):
            kind = joint.get("type", defaults.get(joint.get("class", current), "hinge"))
            if kind != "free":
                joints.append(kind)
        for body in node.findall("body"):
            read_body(body, current)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        read_body(worldbody, "main")
    return joints


def scan_local_assets(root):
    """External assets require asset.json semantic metadata; do not guess labels."""
    assets = []
    for path in sorted(Path(root).rglob("asset.json")):
        item = read_json(path)
        if not item.get("category") or not item.get("model"):
            continue
        model_path = (path.parent / item["model"]).resolve()
        suffix = model_path.suffix.lower()
        if suffix not in {".xml", ".mjcf", ".urdf", ".usd", ".usda", ".usdc"}:
            raise ValueError(f"Unsupported indexed model format: {model_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing indexed model: {model_path}")
        root_xml = None if suffix in {".usd", ".usda", ".usdc"} else ET.parse(model_path).getroot()
        if root_xml is None:
            if "joints" not in item or not isinstance(item["joints"], list):
                raise ValueError(f"USD assets need declared joints metadata (including [] for static/conveyor assets): {path}")
            joints = item["joints"]
        elif suffix == ".urdf":
            mapping = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}
            joints = [mapping[j.get("type")] for j in root_xml.findall("joint") if j.get("type") in mapping]
        else:
            # Native MJCF fixtures must be expanded; include-only documents are rejected.
            if any(True for _ in root_xml.iter("include")):
                raise ValueError(f"Expand MJCF includes before indexing {model_path}")
            joints = mjcf_joint_types(root_xml)
        assets.append({**item, "id": item.get("id", str(path.parent)),
                       "source": "local", "model": str(model_path),
                       "joints": joints, "parametric": False,
                       "topology_source": "declared; verify in USD" if root_xml is None else "parsed model",
                       "appearance_score": None, "appearance_evaluated": False})
    return assets


def select_assets(request, candidates):
    required = Counter(request.get("joint_types", []))
    geometry = request.get("geometry", {})
    accepted, rejected = [], []
    for candidate in candidates:
        reasons, score = [], 0
        if candidate.get("category") != request["category"]:
            reasons.append("category_mismatch")
        if Counter(candidate.get("joints", [])) != required:
            reasons.append("joint_topology_mismatch")
        if request.get("behavior") and candidate.get("behavior") not in (None, request["behavior"]):
            reasons.append("behavior_mismatch")
        if (request.get("behavior") == "deformable" or request.get("deformable")) and candidate.get("behavior") != "deformable":
            if "behavior_mismatch" not in reasons:
                reasons.append("behavior_mismatch")
        panel = geometry.get("panel_type")
        if panel and candidate.get("panel_type") not in (panel, None):
            reasons.append("panel_shape_mismatch")
        if candidate["source"] == "local" and not candidate.get("license"):
            reasons.append("missing_source_license_metadata")
        if reasons:
            rejected.append({"id": candidate["id"], "reasons": reasons})
            continue
        score += 50
        score += 25 if candidate.get("panel_type") == panel and panel else 0
        score += 15 if candidate.get("parametric") else 0
        accepted.append({**candidate, "structure_score": score,
                         "review_required": True,
                         "evidence": ["category and joint topology compatible", "appearance and metric fit need review"]})
    accepted.sort(key=lambda item: (-item["structure_score"], item["id"]))
    return {"schema_version": 1, "object_id": request["id"],
            "route": "reuse_review" if accepted else "reconstruct",
            "candidates": accepted, "rejected": rejected,
            "photo_embedding_search": "not implemented",
            "selection_basis": "explicit object attributes; structural score is not a visual-similarity score"}

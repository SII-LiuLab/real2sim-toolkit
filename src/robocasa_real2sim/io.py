import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_request(path):
    request = read_json(path)
    if not isinstance(request.get("id"), str) or not request["id"]:
        raise ValueError("request.id must be a nonempty string")
    if not request.get("category"):
        raise ValueError("request.category is required")
    for capture in request.get("captures", []):
        if not capture.get("state"):
            raise ValueError("Every photo needs an explicit articulation state")
        image_path = Path(capture["path"])
        if not image_path.is_absolute():
            image_path = Path(path).resolve().parent / image_path
        capture["path"] = str(image_path.resolve())
        if capture.get("mask"):
            mask_path = Path(capture["mask"])
            if not mask_path.is_absolute():
                mask_path = Path(path).resolve().parent / mask_path
            capture["mask"] = str(mask_path.resolve())
    reference = request.get("appearance", {}).get("reference")
    if reference:
        reference_path = Path(reference["path"])
        if not reference_path.is_absolute():
            reference_path = Path(path).resolve().parent / reference_path
        reference["path"] = str(reference_path.resolve())
    for part in request.get("geometry", {}).get("parts", []):
        for key in ("mesh",):
            if part.get(key):
                value = Path(part[key])
                part[key] = str((Path(path).resolve().parent / value).resolve())
        if part.get("collision_meshes"):
            part["collision_meshes"] = [str((Path(path).resolve().parent / value).resolve()) for value in part["collision_meshes"]]
        appearance = part.get("appearance", {})
        if appearance.get("texture"):
            appearance["texture"] = str((Path(path).resolve().parent / appearance["texture"]).resolve())
    return request

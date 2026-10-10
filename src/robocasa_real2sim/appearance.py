"""Rank local surface textures against an explicitly selected photograph patch."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def surface_features(image):
    image = image.convert("RGB").copy()
    image.thumbnail((512, 512))
    pixels = np.asarray(image, dtype=float) / 255
    gray = pixels @ np.array([0.299, 0.587, 0.114])
    dx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0
    dy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0
    return {"median_rgb": np.median(pixels.reshape(-1, 3), axis=0).tolist(),
            "contrast": float(np.percentile(gray, 90) - np.percentile(gray, 10)),
            "grain_direction": (dx - dy) / (dx + dy + 1e-9)}


def rank_textures(request, paths=None, limit=5):
    appearance = request.get("appearance", {})
    reference = appearance.get("reference")
    if not reference:
        return {"status": "not_requested", "candidates": []}
    bbox = reference.get("bbox", [0, 0, 1, 1])
    if (len(bbox) != 4 or not np.isfinite(bbox).all() or
            not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
        raise ValueError("appearance.reference.bbox must be normalized [left, top, right, bottom]")
    try:
        with Image.open(reference["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            box = tuple(round(v * image.size[i % 2]) for i, v in enumerate(bbox))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                raise ValueError("Appearance reference patch is too small")
            features = surface_features(image.crop(box))
    except OSError as error:
        return {"status": "reference_unavailable", "reference": reference,
                "error": str(error), "candidates": [], "review_required": True}
    if paths is None:
        from robocasa.models import assets_root
        paths = sorted((Path(assets_root) / "textures" / "wood").glob("*.png"))
    candidates, rejected = [], []
    for path in paths:
        try:
            with Image.open(path) as source:
                other = surface_features(source)
        except (OSError, ValueError) as error:
            rejected.append({"texture": str(path), "reason": str(error)})
            continue
        differences = {
            "color": float(np.abs(np.array(features["median_rgb"]) - other["median_rgb"]).mean()),
            "contrast": abs(features["contrast"] - other["contrast"]),
            "grain_direction": abs(features["grain_direction"] - other["grain_direction"]) / 2,
        }
        distance = sum(differences[key] * weight for key, weight in
                       (("color", 0.7), ("contrast", 0.2), ("grain_direction", 0.1)))
        candidates.append({"texture": Path(path).resolve().as_posix(),
                           "score": round(100 * (1 - distance), 2),
                           "differences": differences, "features": other})
    candidates.sort(key=lambda row: (-row["score"], row["texture"]))
    return {"status": "ranked" if candidates else "no_textures",
            "method": "median RGB, luminance contrast and directional gradient; heuristic score",
            "reference": reference, "reference_features": features,
            "candidates": candidates[:limit], "evaluated_count": len(candidates), "rejected": rejected,
            "review_required": True, "appearance_calibrated": False,
            "limitations": ["Lighting and photograph perspective affect scores.",
                            "Native UV orientation must be checked in the rendered model.",
                            "This ranks surface materials, not object identity or geometry."]}

"""Inspect real images; never combine different articulation states for SfM."""

from collections import defaultdict
from hashlib import sha256

import numpy as np
from PIL import Image, ImageOps


def inspect_captures(request, blur_threshold=20.0):
    images, seen = [], set()
    groups = defaultdict(list)
    for capture in request.get("captures", []):
        entry = dict(capture)
        entry.setdefault("part", "assembly")
        try:
            with Image.open(capture["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                digest = sha256(np.asarray(image).tobytes()).hexdigest()
                gray = image.convert("L")
                gray.thumbnail((1024, 1024))
                pixels = np.asarray(gray, dtype=float)
                laplace = -4 * pixels[1:-1, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:] + pixels[:-2, 1:-1] + pixels[2:, 1:-1]
                sharpness = float(np.var(laplace)) if laplace.size else 0.0
            duplicate_key = (entry["state"], entry["part"], digest)
            duplicate = duplicate_key in seen
            seen.add(duplicate_key)
            entry.update(width=width, height=height, pixel_sha256=digest,
                         sharpness=sharpness, duplicate=duplicate,
                         accepted=not duplicate and min(width, height) >= 64)
            entry["review_reasons"] = ["possible_blur"] if sharpness < blur_threshold else []
            if entry.get("mask"):
                with Image.open(entry["mask"]) as mask:
                    if mask.size != (width, height):
                        raise ValueError("Mask dimensions differ from the EXIF-oriented image")
            if entry["accepted"]:
                groups[(entry["state"], entry["part"])].append(entry["path"])
        except (OSError, ValueError) as error:
            entry.update(accepted=False, error=str(error))
        images.append(entry)
    return {
        "schema_version": 1, "object_id": request["id"], "images": images,
        "groups": [
            {"state": state, "part": part, "images": paths,
             "count": len(paths), "minimum_views_available": len(paths) >= 3}
            for (state, part), paths in sorted(groups.items())
        ],
        "notes": ["Blur is advisory; exact duplicates are excluded within each state/part.",
                  "Three unique views is only a minimum; successful pose recovery and overlap must be checked."],
    }

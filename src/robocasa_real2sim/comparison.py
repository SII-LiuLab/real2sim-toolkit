"""Explicit silhouette measurements and review images for agent iterations.

Image similarity cannot establish physical calibration. Masks must be supplied
by the caller; colors in an RGB render are never treated as segmentation.
"""

from hashlib import sha256
import math
from pathlib import Path
import tempfile

from .io import write_json


def mask_metrics(reference_mask, rendered_mask):
    """Measure aligned, nonempty 2D binary masks without resizing or registration."""
    import numpy as np

    masks = []
    for value in (reference_mask, rendered_mask):
        value = np.asarray(value)
        if value.ndim != 2 or not value.size:
            raise ValueError("Masks must be nonempty 2D arrays")
        if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError("Masks must contain numeric or boolean values")
        if not np.isfinite(value).all():
            raise ValueError("Masks must contain finite values")
        if not np.isin(value, (0, 1)).all():
            raise ValueError("mask_metrics expects binary masks containing only 0 and 1")
        mask = value.astype(bool)
        if not mask.any():
            raise ValueError("Masks must each have a nonempty foreground")
        masks.append(mask)
    reference, rendered = masks
    if reference.shape != rendered.shape:
        raise ValueError("Mask dimensions differ; align cameras and resolution explicitly")

    def describe(mask):
        y, x = np.nonzero(mask)
        return {"area_px": int(mask.sum()), "centroid_xy_px": [float(x.mean()), float(y.mean())],
                "bbox_xyxy_px": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]}

    ref, sim = describe(reference), describe(rendered)
    bbox_ref, bbox_sim = ref["bbox_xyxy_px"], sim["bbox_xyxy_px"]
    centroid_distance = float(np.linalg.norm(np.array(sim["centroid_xy_px"]) - ref["centroid_xy_px"]))
    return {
        "silhouette_iou": float(np.logical_and(reference, rendered).sum() / np.logical_or(reference, rendered).sum()),
        "reference": ref, "rendered": sim,
        "area_ratio": sim["area_px"] / ref["area_px"],
        "bbox_width_ratio": (bbox_sim[2] - bbox_sim[0]) / (bbox_ref[2] - bbox_ref[0]),
        "bbox_height_ratio": (bbox_sim[3] - bbox_sim[1]) / (bbox_ref[3] - bbox_ref[1]),
        "centroid_distance_px": centroid_distance,
        "centroid_distance_normalized": centroid_distance / math.hypot(*reference.shape),
    }


def _load_mask(path, threshold):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        mask = np.asarray(ImageOps.exif_transpose(source))
    if mask.ndim == 3:
        # Grayscale replicated to RGB is unambiguous. An RGB label map is not.
        if mask.shape[2] not in (3, 4) or not np.all(mask[..., :3] == mask[..., :1]):
            raise ValueError("Masks must be grayscale; convert labeled/color masks explicitly")
        mask = mask[..., 0]
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("Masks must be finite grayscale images")
    return mask > threshold


def compare_images(reference, rendered, output_dir, *, reference_mask=None,
                   rendered_mask=None, mask_threshold=0, min_iou=None):
    """Write a side-by-side review and optional silhouette metrics.

    Both masks are required for metrics. Their nonzero pixels are foreground
    by default. Thresholding, if requested, is applied to masks only. Images
    and masks must have equal, EXIF-oriented resolution for measured comparison.
    Without masks this returns ``review_required`` and never a successful match.
    ``min_iou`` evaluates only silhouette agreement, never reconstruction success.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "comparison.json"
    inputs = (reference, rendered, reference_mask, rendered_mask)
    if any(path is not None and Path(path).resolve() == report_path for path in inputs):
        raise ValueError("Comparison inputs must not be the output report path")
    report = {"schema_version": 1, "status": "failed", "review_required": True,
              "physical_match_validated": False, "metrics": {}, "artifacts": {}}
    write_json(report_path, report)
    try:
        if bool(reference_mask is not None) != bool(rendered_mask is not None):
            raise ValueError("Provide both reference_mask and rendered_mask")
        mask_threshold = float(mask_threshold)
        if not math.isfinite(mask_threshold) or mask_threshold < 0:
            raise ValueError("mask_threshold must be finite and nonnegative")
        if min_iou is not None:
            min_iou = float(min_iou)
            if not math.isfinite(min_iou) or not 0 <= min_iou <= 1:
                raise ValueError("min_iou must be finite and between 0 and 1")
            if reference_mask is None:
                raise ValueError("min_iou requires explicit masks")
        images = []
        for path in (reference, rendered):
            with Image.open(path) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        ref, sim = images
        report.update(reference=str(Path(reference).resolve()), rendered=str(Path(rendered).resolve()),
                      image_sizes={"reference": list(ref.size), "rendered": list(sim.size)},
                      pixel_sha256={"reference": sha256(ref.tobytes()).hexdigest(),
                                    "rendered": sha256(sim.tobytes()).hexdigest()})
        masks = None
        if reference_mask is not None:
            if ref.size != sim.size:
                raise ValueError("Image dimensions differ; align cameras and resolution explicitly")
            masks = [_load_mask(path, mask_threshold) for path in (reference_mask, rendered_mask)]
            if any(mask.shape != (ref.height, ref.width) for mask in masks):
                raise ValueError("Mask dimensions must match the oriented images")
            report["metrics"] = mask_metrics(*masks)
            report.update(reference_mask=str(Path(reference_mask).resolve()),
                          rendered_mask=str(Path(rendered_mask).resolve()), mask_threshold=mask_threshold)
            report["status"] = "measured"
            if min_iou is not None:
                report.update(silhouette_passed=report["metrics"]["silhouette_iou"] >= min_iou,
                              min_iou=min_iou)
        else:
            report["status"] = "review_required"
        # Never write over an input, even when called again on an old artifact.
        artifacts = Path(tempfile.mkdtemp(prefix="review-", dir=output))
        width, height = ref.width + sim.width, max(ref.height, sim.height)
        sheet = Image.new("RGB", (width, height + 24), "white")
        sheet.paste(ref, (0, 24))
        sheet.paste(sim, (ref.width, 24))
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), "Reference", fill="black")
        draw.text((ref.width + 4, 4), "Rendered", fill="black")
        path = artifacts / "side_by_side.png"
        sheet.save(path)
        report["artifacts"]["side_by_side"] = str(path)
        if masks is not None:
            overlay = np.asarray(ref).copy()
            # Green = reference only, magenta = render only, white = intersection.
            overlay[masks[0] & ~masks[1]] = [0, 210, 80]
            overlay[masks[1] & ~masks[0]] = [230, 0, 180]
            overlay[masks[0] & masks[1]] = [255, 255, 255]
            path = artifacts / "silhouette_overlay.png"
            Image.fromarray(overlay).save(path)
            report["artifacts"]["silhouette_overlay"] = str(path)
            report["overlay_legend"] = {"reference_only": "green", "rendered_only": "magenta", "overlap": "white"}
        report["notes"] = [
            "Silhouette metrics use explicitly supplied masks, with no automatic alignment or resizing.",
            "Visual agreement does not validate material, contact, dynamics or physical calibration.",
        ]
        write_json(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise

"""Render the exported MJCF itself for a repeatable visual review."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .validation import load_mjcf


def render_preview(model_path, output):
    import mujoco
    model = load_mjcf(model_path)
    if model.njnt > 2 or any(kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                             for kind in model.jnt_type):
        raise ValueError("Preview supports exported fixtures with at most two hinge/slide joints")
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 640
    model.vis.headlight.ambient[:] = 0.5
    camera = mujoco.MjvCamera()
    camera.lookat[:] = model.stat.center
    camera.distance = model.stat.extent * 1.25
    camera.azimuth, camera.elevation = 110, -12
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.sitegroup[:] = 0
    target = np.where(model.jnt_range[:, 0] < 0, model.jnt_range[:, 0], model.jnt_range[:, 1]) * 0.95
    sheet = Image.new("RGB", (1280, 676), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    with mujoco.Renderer(model, height=640, width=640) as renderer:
        for index, factor in enumerate((0, 1)):
            data.qpos[model.jnt_qposadr] = target * factor
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            sheet.paste(Image.fromarray(renderer.render()), (index * 640, 36))
            draw.text((index * 640 + 16, 12), "Exported MJCF - " + ("closed" if index == 0 else "open"), fill="white")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"preview": str(output.resolve()), "model": str(Path(model_path).resolve()),
            "rendered_states": ["closed", "open"], "real_world_calibrated": False}

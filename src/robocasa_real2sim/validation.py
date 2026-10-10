from itertools import product
from pathlib import Path

import numpy as np

from .io import write_json


def load_mjcf(path):
    import mujoco
    path = Path(path)
    # VFS avoids the native Windows filename issue for Chinese workspace paths.
    assets = {file.relative_to(path.parent).as_posix(): file.read_bytes()
              for file in (path.parent / "assets").rglob("*") if file.is_file()}
    return mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"), assets=assets)


def validate_asset(path):
    report_path = Path(path).parent / "validation.json"
    write_json(report_path, {"passed": False, "status": "running"})
    try:
        return _validate_asset(path)
    except (Exception, KeyboardInterrupt) as error:
        write_json(report_path, {"passed": False,
                                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                "error": str(error)})
        raise


def _validate_asset(path):
    import mujoco
    model = load_mjcf(path)
    data = mujoco.MjData(model)
    if model.njnt > 2 or any(j not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in model.jnt_type):
        raise ValueError("This native fixture validator supports at most two hinge/slide joints")
    qaddr, daddr = model.jnt_qposadr, model.jnt_dofadr
    target = np.where(model.jnt_range[:, 0] < 0, model.jnt_range[:, 0], model.jnt_range[:, 1]) * 0.95
    worst, poses = 0.0, 0
    for factors in product(np.linspace(0, 1, 19), repeat=model.njnt):
        data.qpos[qaddr] = target * factors
        mujoco.mj_forward(model, data)
        worst = max(worst, max([0.0] + [-float(c.dist) for c in data.contact]))
        poses += 1
    mujoco.mj_resetData(model, data)
    slide = model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE
    kp, kd, limit = np.where(slide, 40, 8), np.where(slide, 5, 0.8), np.where(slide, 20, 2)
    hold_spans = []
    for goal in (target, np.zeros_like(target)):
        final = []
        for i in range(round(6.0 / model.opt.timestep)):
            data.qfrc_applied[daddr] = np.clip(kp * (goal - data.qpos[qaddr]) - kd * data.qvel[daddr], -limit, limit)
            mujoco.mj_step(model, data)
            worst = max(worst, max([0.0] + [-float(c.dist) for c in data.contact]))
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite dynamics")
            if i >= round(5.0 / model.opt.timestep):
                final.append(data.qpos[qaddr].copy())
        span = np.ptp(final, axis=0)
        hold_spans.append(span.tolist())
        if not np.allclose(data.qpos[qaddr], goal, atol=0.005) or np.any(span > 0.002):
            raise RuntimeError("Joint cannot reach and hold the open/closed target")
    if worst > 0.00001 or np.any(data.warning.number):
        raise RuntimeError(f"Physics validation failed: penetration={worst} m")
    report = {"passed": True, "status": "passed", "scope": "standalone fixture geometry and dynamics",
              "joint_count": model.njnt, "poses_checked": poses,
              "max_self_penetration_m": worst, "hold_span_radians_or_m": hold_spans,
              "validation_controller": {"kp": kp.tolist(), "kd": kd.tolist(), "force_or_torque_limit": limit.tolist()},
              "real_world_physics_calibrated": False,
              "robot_interaction_tested": False}
    write_json(Path(path).parent / "validation.json", report)
    return report

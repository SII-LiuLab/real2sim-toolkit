"""Re-render captured native RSS states without advancing a physics solver.

Imports of RoboSimStudio remain lazy. Geometry and particle order must match
the recording; materials, lighting, cameras and static scenery may be revised.
An explicit callback can supply missing rigid poses for an older particle-only
recording. That is declared state reconstruction, not recorded robot state.
"""

from collections.abc import Mapping
from hashlib import sha256
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .rss_compat import offline_renderer_type, resolve_camera, scene_directory


_CHANNELS = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")


def _digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key].copy() for key in file.files}


def _shapes(model):
    return {"particle_q": (int(model.particle_count), 3),
            "particle_qd": (int(model.particle_count), 3),
            "body_q": (int(model.body_count), 7),
            "body_qd": (int(model.body_count), 6),
            "joint_q": (int(model.joint_coord_count),),
            "joint_qd": (int(model.joint_dof_count),)}


def _state_array(key, value, shape):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{key} shape {array.shape} does not match model {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{key} must contain finite numeric values")
    if key == "body_q" and len(array):
        norms = np.linalg.norm(array[:, 3:7], axis=1)
        if not np.allclose(norms, 1., rtol=0., atol=1.e-3):
            raise ValueError("body_q contains non-unit pose quaternions")
    return array


def _geometry(model, saved):
    result = {}
    count = int(model.particle_count)
    if count:
        if "particle_q" not in saved:
            raise ValueError("model_geometry must contain reference particle_q")
        reference = _state_array("particle_q", saved["particle_q"], (count, 3))
        current = model.particle_q.numpy()
        if not np.allclose(reference, current, rtol=1.e-6, atol=1.e-7):
            raise ValueError("Rebuilt model reference particle geometry/order differs from recording")
        result["reference_particles"] = {"count": count, "matched": True,
            "maximum_absolute_difference_m": float(np.max(abs(reference-current)))}
    for key, width, count_name in (("tri_indices", 3, "tri_count"),
                                   ("tet_indices", 4, "tet_count")):
        expected_count = int(getattr(model, count_name))
        recorded = saved.get(key)
        if recorded is None:
            if expected_count:
                raise ValueError(f"model_geometry is missing {key}")
            result[key] = {"count": 0, "matched": True}
            continue
        recorded = np.asarray(recorded)
        if recorded.shape != (expected_count, width) or not np.issubdtype(recorded.dtype, np.integer):
            raise ValueError(f"Recorded {key} shape/type does not match rebuilt model")
        current_array = getattr(model, key, None)
        actual = (current_array.numpy() if current_array is not None
                  else np.empty((0, width), dtype=np.int32))
        if not np.array_equal(recorded, actual):
            raise ValueError(f"Rebuilt {key} or its vertex order differs from recording")
        if recorded.size and (recorded.min() < 0 or recorded.max() >= count):
            raise ValueError(f"{key} contains invalid particle IDs")
        result[key] = {"count": expected_count, "matched": True}
    # Static background shape counts may change during look development. Rigid
    # body/joint order cannot: those rows are how saved state addresses bodies.
    for key in ("body_label", "joint_label"):
        if key in saved:
            current = np.asarray(getattr(model, key), dtype=str)
            recorded = np.asarray(saved[key], dtype=str)
            if not np.array_equal(recorded, current):
                raise ValueError(f"Rebuilt {key} ordering differs from recording")
            result[key] = {"count": len(current), "matched": True}
        else:
            result[key] = {"matched": None, "reason": "not recorded"}
    return result


def _callback_metadata(callback, provenance):
    if callback is None:
        return None
    result = {"module": getattr(callback, "__module__", None),
              "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
              "provenance": provenance,
              "kind": "explicit reconstruction of missing state; not recorded rigid-body poses"}
    try:
        source = inspect.getsourcefile(callback)
        if source and Path(source).is_file():
            result.update(source_path=str(Path(source).resolve()), source_sha256=_digest(source))
        result["callable_source_sha256"] = sha256(inspect.getsource(callback).encode()).hexdigest()
    except (OSError, TypeError):
        result["source_available"] = False
    return result


def render_trajectory(runtime, trajectory_path, model_geometry_path, output_dir,
                      camera=None, every=2, record=True, *, state_callback=None,
                      callback_provenance=None, fps=None):
    """Render real captured states with native OfflineRenderer/StateReplay.

    ``state_callback(runtime, time_s, source_frame_index)`` may return a mapping
    of *missing* state channels in current model order. It must not step or
    mutate the live runtime. In particular, a robot scene with particle-only
    capture is rejected unless this callback explicitly supplies ``body_q`` or
    full ``joint_q``. The latter is converted to poses by native Newton FK,
    without a solver step. Existing captured channels cannot be overridden.

    ``every`` subsamples existing frames and includes the final frame. PNGs and
    exact timestamps are authoritative; optional MP4 is constant frame rate.
    Failures return ``passed=False`` and overwrite report.json with the error.
    A successful render does not validate reconstruction or physical behavior.
    """
    out = Path(output_dir).expanduser().resolve()
    trajectory_path = Path(trajectory_path).expanduser().resolve()
    model_geometry_path = Path(model_geometry_path).expanduser().resolve()
    if out in {trajectory_path.parent, model_geometry_path.parent}:
        raise ValueError("Use a separate render output directory to preserve physical run evidence")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    started = time.perf_counter()
    report = {"status": "running", "passed": False, "physics_steps": 0,
              "trajectory": str(trajectory_path), "model_geometry": str(model_geometry_path),
              "report_path": str(report_path), "captures": [],
              "reconstruction_validated": False, "recorded_state_modified": False}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    save_report()  # Supersede success from an earlier attempt immediately.
    offline = None
    renderer = None
    writer = None
    original_state = {}
    sim = None
    start_counter = None
    try:
        runtime._open()
        if isinstance(every, bool) or int(every) != every or every < 1:
            raise ValueError("every must be a positive integer")
        every = int(every)
        if state_callback is not None and not callable(state_callback):
            raise TypeError("state_callback must be callable")
        # Validate provenance serializability before an expensive render.
        metadata = _callback_metadata(state_callback, callback_provenance)
        report["state_callback"] = json.loads(json.dumps(metadata, allow_nan=False))
        sim = runtime.sim
        sim.sync()
        start_counter = (int(sim.step_count), float(sim.time))
        for key in _CHANNELS:
            value = getattr(sim.state, key, None)
            if value is not None:
                original_state[key] = value.numpy().copy()
        model = sim.model
        shapes = _shapes(model)
        arrays = _load(trajectory_path)
        geometry = _load(model_geometry_path)
        report.update(trajectory_sha256=_digest(trajectory_path),
                      model_geometry_sha256=_digest(model_geometry_path),
                      render_scene=str(runtime.path), render_scene_sha256=_digest(runtime.path),
                      renderer=runtime.render_report(),
                      adapter_sha256=_digest(__file__))
        if "times_s" not in arrays:
            raise ValueError("trajectory must contain exact times_s timestamps")
        times = np.asarray(arrays["times_s"], dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError("times_s must be a nonempty finite one-dimensional array")
        if len(times) > 1 and not np.all(np.diff(times) > 0.):
            raise ValueError("trajectory timestamps must be strictly increasing")
        count = len(times)
        channels = {key: arrays[key] for key in _CHANNELS if key in arrays}
        for key, values in channels.items():
            if values.shape != (count, *shapes[key]):
                raise ValueError(f"Recorded {key} shape {values.shape} does not match {(count, *shapes[key])}")
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"Recorded {key} contains non-finite/non-numeric values")
            if key == "body_q":
                _state_array(key, values.reshape(-1, 7), (count*int(model.body_count), 7))
        if int(model.particle_count) and "particle_q" not in channels:
            raise ValueError("Particle scene requires full recorded particle_q; no static-cloth fallback")
        rigid_recorded = "body_q" in channels or "joint_q" in channels
        if int(model.body_count) and not rigid_recorded and state_callback is None:
            raise ValueError("Trajectory lacks robot/rigid poses. Supply recorded body_q/joint_q or an explicit trusted state_callback")
        report["geometry_validation"] = _geometry(model, geometry)
        if int(model.body_count) and rigid_recorded:
            needed = "body_label" if "body_q" in channels else "joint_label"
            if needed not in geometry:
                raise ValueError(f"Recorded rigid poses require {needed} in model_geometry to verify row ordering")
        steps = arrays.get("steps")
        if steps is not None and (steps.shape != (count,) or not np.issubdtype(steps.dtype, np.integer)):
            raise ValueError("Recorded steps must be one integer per timestamp")
        indices = list(range(0, count, every))
        if indices[-1] != count-1:
            indices.append(count-1)
        selected_times = times[indices]
        if fps is None:
            fps = (1./float(np.median(np.diff(selected_times))) if len(indices) > 1
                   else float(runtime.scene.physics.control_hz)/every)
        if isinstance(fps, bool) or not np.isfinite(float(fps)) or float(fps) <= 0.:
            raise ValueError("fps must be positive and finite")
        fps = float(fps)
        spec = resolve_camera(sim, camera)
        report.update(source_frame_count=count, every=every,
            recorded_channels=sorted(channels), ignored_arrays=sorted(set(arrays)-set(channels)-{"steps", "times_s"}),
            source_frame_indices=indices, selected_times_s=selected_times.tolist(),
            camera={"name": spec.name, "width": int(spec.width), "height": int(spec.height),
                    "intrinsics": spec.intrinsics.tolist(),
                    "camera_local_matrix": spec.world_matrix.tolist(),
                    "parent_body": int(spec.parent_body)},
            rigid_state_source=("recorded" if rigid_recorded else "explicit_callback" if int(model.body_count) else "none"))
        from PIL import Image
        OfflineRenderer = offline_renderer_type()

        with scene_directory(runtime.path):
            renderer = sim.make_renderer(runtime.renderer_name, cameras=[spec], channels=["rgb"])
        offline = OfflineRenderer(model, renderer, strict=True)
        # RTX caches planar UVs from its FIRST particle shape. Stage against the
        # captured reference geometry, never an already-folded selected frame.
        reference = {"particle_q": geometry["particle_q"]} if int(model.particle_count) else {}
        rest_state = offline.replay_state.write(reference)
        with scene_directory(runtime.path):
            renderer.render(rest_state)
        report["reference_geometry_warmup"] = True
        if record:
            import imageio.v2 as imageio
            video = out / "replay.mp4"
            writer = imageio.get_writer(str(video), fps=fps, quality=7, macro_block_size=1)
            report.update(video=str(video), video_fps=fps,
                video_timing="constant_frame_rate; selected_times_s/captures retain exact source simulation times")
        callback_channels = set()
        for index in indices:
            step = {key: value[index] for key, value in channels.items()}
            if state_callback is not None:
                with scene_directory(runtime.path):
                    extra = state_callback(runtime, float(times[index]), index)
                if not isinstance(extra, Mapping):
                    raise TypeError("state_callback must return a mapping of missing state arrays")
                unknown = set(extra)-set(_CHANNELS)
                if unknown:
                    raise ValueError(f"Unsupported state_callback channels: {sorted(unknown)}")
                overlap = set(extra) & set(step)
                if overlap:
                    raise ValueError(f"Callback cannot override captured state channels: {sorted(overlap)}")
                for key, value in extra.items():
                    step[key] = _state_array(key, value, shapes[key])
                callback_channels.update(extra)
                if (int(sim.step_count), float(sim.time)) != start_counter:
                    raise RuntimeError("state_callback advanced live physics; offline rendering forbids stepping")
            if int(model.body_count) and not ("body_q" in step or
                    ("joint_q" in step and int(model.joint_coord_count))):
                raise ValueError("Every rendered frame requires recorded or explicit reconstructed rigid state")
            # Native write honours joint_q but not joint_qd. Set the latter on
            # this independent replay state before native FK, if supplied.
            if "joint_qd" in step:
                offline.replay_state.state.joint_qd.assign(step["joint_qd"])
            with scene_directory(runtime.path):
                frame_set = next(offline.replay([step]))
            rgb = np.asarray(frame_set[spec.name]["rgb"])
            if rgb.shape != (spec.height, spec.width, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Renderer returned unexpected RGB format {rgb.shape}/{rgb.dtype}")
            target = out / f"frame_{index:06d}.png"
            Image.fromarray(rgb).save(target)
            if writer is not None:
                writer.append_data(rgb)
            capture = {"path": str(target), "source_frame_index": index,
                       "time_s": float(times[index]), "camera": spec.name}
            if steps is not None:
                capture["step"] = int(steps[index])
            report["captures"].append(capture)
        report["callback_channels"] = sorted(callback_channels)
        report["status"] = "completed"
        report["passed"] = True
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        for resource in (writer, offline if offline is not None else renderer):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report.update(status="failed", passed=False,
                                  cleanup_error=f"{type(error).__name__}: {error}")
        if sim is not None and start_counter is not None:
            sim.sync()
            report["physics_steps"] = int(sim.step_count)-start_counter[0]
            unchanged = (int(sim.step_count), float(sim.time)) == start_counter
            for key, original in original_state.items():
                unchanged = unchanged and np.array_equal(original, getattr(sim.state, key).numpy())
            report["runtime_state_unchanged"] = bool(unchanged)
            if not unchanged:
                report.update(status="failed", passed=False,
                    error="Offline render/callback changed the live runtime; callback must be pure and must not step")
        report["seconds"] = time.perf_counter()-started
        report["rendered_frames"] = len(report["captures"])
        save_report()
    return report

"""Optional headless RoboSimStudio Scene/Sim adapter for an agent harness.

Scene files are trusted Python programs, not a sandbox. They must bind ``scene``
and may bind ``BUILD``, ``routine(sim)`` or ``routines``. A routine advances the
simulator itself and yields once per control step, matching RoboSimStudio.
Simulator imports stay lazy so CPU tooling works without this optional backend.
"""

from hashlib import sha256
import inspect
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .rss_compat import (
    install_renderer_bridge as _renderer_compatibility,
    resolve_camera,
    rss_module as _rss,
    scene_directory as _scene_directory,
    suppress_scene_viewers,
)


FLEXIBLE_DOMAINS = ("cloth", "soft", "rod")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _mesh_strain(rest, triangles, tetrahedra, positions, times):
    """Diagnostics on actual mesh edges; no assumed grid layout or pass limit."""
    pieces = []
    for indices, pairs in ((triangles, ((0, 1), (1, 2), (2, 0))),
                           (tetrahedra, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))):
        if indices.size:
            pieces.extend(indices[:, pair] for pair in pairs)
    if not pieces or not len(positions):
        return {"status": "unavailable", "reason": "No triangle/tetrahedral mesh edges"}
    edges = np.unique(np.sort(np.concatenate(pieces), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    valid = lengths > 1e-12
    degenerate = int((~valid).sum())
    edges, lengths = edges[valid], lengths[valid]
    if not len(edges):
        return {"status": "unavailable", "reason": "Degenerate rest edges", "degenerate_edges": degenerate}
    tensile, compressive = 0.0, 0.0
    worst = None
    for frame, points in enumerate(positions):
        strain = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1) / lengths - 1
        index = int(np.argmax(strain))
        if float(strain[index]) > tensile:
            tensile = float(strain[index])
            worst = {"vertex_ids": edges[index].tolist(), "time_s": float(times[frame]),
                     "rest_length_m": float(lengths[index]),
                     "length_m": float(lengths[index] * (1 + tensile))}
        compressive = max(compressive, float(-strain.min()))
    return {"status": "measured", "edge_count": len(edges), "degenerate_rest_edges": degenerate,
            "max_tensile_strain": tensile, "max_compressive_strain": compressive,
            "worst_tensile_edge": worst, "physical_match_validated": False,
            "notes": ["Dimensionless length/rest_length - 1 on actual triangle/tetrahedral edges.",
                      "A finite state or an execution pass does not validate these deformation magnitudes."]}


def _newton_physics(scene, rss, device):
    """Set domain defaults; refuse per-asset overrides which defeat the choice."""
    original = scene.physics
    default = original.backend_for("*")
    if default is None or default.solver != "vbd":
        default = rss.backend("vbd")
    physics = original
    for domain in FLEXIBLE_DOMAINS:
        # Respect staged VBD parameters (iterations, self contact, buffers).
        # Replace only backends which would change the requested engine.
        authored = original.backend_for(domain)
        chosen = authored if authored is not None and authored.solver == "vbd" else default
        if str(device) == "cpu":
            chosen = chosen.with_(particle_enable_tile_solve=False)
        physics = physics.with_backend(domain, chosen)
    # A per-asset backend takes precedence over the domain table. Refuse it
    # rather than claiming Newton while building an entity under another solver.
    for spec in scene.request.entities:
        if (getattr(spec, "entity", "") == "soft" and
                any(getattr(spec, field, None) is not None for field in ("anchored", "rigid"))):
            raise ValueError(f"{spec.name!r}: native SoftSpec.anchored/rigid are host-engine features; this RoboSimStudio version does not apply them in Newton VBD")
        requirements = getattr(spec, "sim", None)
        if requirements is not None and requirements.domain in FLEXIBLE_DOMAINS:
            override = requirements.backend
            if override is not None and override.solver != "vbd":
                raise ValueError(f"{spec.name!r} overrides its flexible backend to {override.solver!r}; remove that override or choose 'vbd'")
            if override is not None and str(device) == "cpu" and override.get("particle_enable_tile_solve", False):
                raise ValueError(f"{spec.name!r}: particle_enable_tile_solve must be false for CPU")
    scene.physics = physics
    return [{"domain": domain,
             "before": original.backend_for(domain).solver if original.backend_for(domain) else None,
             "after": "vbd"} for domain in FLEXIBLE_DOMAINS]


def load_scene(path, *, device="cpu", build_overrides=None, force_newton=True, renderer="tiled"):
    """Load a declared Scene and compile it once, without launching a viewer.

    ``device`` is explicit: no CUDA-to-CPU fallback. ``BUILD`` is passed to
    Scene.build, with caller overrides taking precedence. Top-level prebuilt
    ``sim`` is rejected because it bypasses backend/device policy. Scene.build
    performs its documented initial warmup/reset; this adapter adds no reset.
    """
    path = Path(path).expanduser().resolve(strict=True)
    from .runtime_env import assert_studio_runtime
    environment = assert_studio_runtime(renderer=renderer)
    rss = _rss()
    started = time.perf_counter()
    namespace = {"__name__": "real2sim_scene", "__file__": str(path)}
    with suppress_scene_viewers(rss):
        with _scene_directory(path):
            exec(compile(path.read_text(), str(path), "exec"), namespace)
    scene = namespace.get("scene")
    if not isinstance(scene, rss.Scene):
        raise TypeError(f"{path.name} must bind scene to rss.Scene")
    if namespace.get("sim") is not None:
        if isinstance(namespace["sim"], rss.Sim):
            namespace["sim"].close()
        raise ValueError("Leave sim unbound: the harness builds scene with explicit backend/device options")
    if namespace.get("routine") is not None and not callable(namespace["routine"]):
        raise TypeError("routine must be callable")
    if namespace.get("routines") is not None and not isinstance(namespace["routines"], dict):
        raise TypeError("routines must be a dict of names to callables")
    build = namespace.get("BUILD", {})
    if not isinstance(build, dict):
        raise TypeError("BUILD must be a dict of Scene.build keyword arguments")
    kwargs = {**build, **(build_overrides or {}), "device": device}
    # Render explicitly through make_renderer on demand. Build-time observation
    # cameras otherwise render every step, even when no evidence is requested.
    if kwargs.get("renderer"):
        raise ValueError("Set the harness renderer option instead of BUILD['renderer']; headless capture is on demand")
    kwargs["renderer"] = ""
    changes = _newton_physics(scene, rss, device) if force_newton else []
    with _scene_directory(path):
        sim = scene.build(**kwargs)
    try:
        bridge = _renderer_compatibility(sim)
    except Exception:
        sim.close()
        raise
    runtime = RSSRuntime(path, namespace, scene, sim, renderer=renderer,
                         backend_changes=changes, build_seconds=time.perf_counter() - started)
    runtime.renderer_bridge = bridge
    runtime.runtime_environment = environment
    try:
        if force_newton:
            for island in sim.pipeline.program.islands:
                if set(island.domains) & set(FLEXIBLE_DOMAINS) and island.backend.solver != "vbd":
                    raise RuntimeError(f"Built flexible island {island.name!r} uses {island.backend.solver!r}, expected vbd")
        return runtime
    except Exception:
        runtime.close()
        raise


class RSSRuntime:
    """One live scene. Operations continue from its current state until close."""

    def __init__(self, path, namespace, scene, sim, *, renderer="tiled", backend_changes=(), build_seconds=0):
        self.path = Path(path)
        self.namespace, self.scene, self.sim = namespace, scene, sim
        self.renderer_name = renderer
        self.renderer_bridge = "native"
        self.runtime_environment = None
        self.backend_changes = list(backend_changes)
        self.build_seconds = build_seconds
        self._renderers = {}
        self._routine = None
        self._routine_name = None
        self._routine_finished = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _open(self):
        if self.closed:
            raise RuntimeError("RoboSimStudio runtime is closed")

    def finite_report(self):
        self._open()
        self.sim.sync()
        checks = {}
        for key in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
            values = getattr(self.sim.state, key, None)
            if values is not None:
                checks[key] = bool(np.isfinite(values.numpy()).all())
        return {"passed": all(checks.values()) and bool(self.sim.state_is_finite()), "arrays": checks}

    def build_report(self):
        self._open()
        import newton
        import warp
        import robosimstudio
        program = self.sim.pipeline.program
        return {"status": "built", "finite": self.finite_report(),
                "scene": str(self.path), "scene_sha256": sha256(self.path.read_bytes()).hexdigest(),
                "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "compatibility_adapter_sha256": sha256(Path(__file__).with_name("rss_compat.py").read_bytes()).hexdigest(),
                "device": str(self.sim.model.device), "seconds": self.build_seconds,
                "versions": {"robosimstudio": robosimstudio.__version__, "newton": newton.__version__, "warp": warp.__version__},
                "runtime_environment": self.runtime_environment,
                "backend_changes": self.backend_changes,
                "islands": [{"name": i.name, "domains": list(i.domains), "entities": list(i.entities),
                             "solver": i.backend.solver, "solver_class": i.solver_class,
                             "params": i.backend.kwargs(), "substeps": i.backend.substeps} for i in program.islands],
                "problems": list(program.problems()), "routines": self.routine_names(),
                "step": int(self.sim.step_count), "time_s": float(self.sim.time),
                "counts": {name: int(getattr(self.sim.model, name)) for name in ("body_count", "particle_count", "tri_count", "tet_count")},
                "render": self.render_report(), "reconstruction_validated": False}

    def render_report(self):
        return {"backend": self.renderer_name, "fidelity": "collider_only" if self.renderer_name == "tiled" else "backend_specific",
                "compatibility_adapter_sha256": sha256(Path(__file__).with_name("rss_compat.py").read_bytes()).hexdigest(),
                "stage_bridge": self.renderer_bridge if self.renderer_name == "rtx" else None,
                "limitations": (["Tiled renders collision geometry and particle surfaces; it does not reproduce appearance/materials.",
                                  "Tiled segmentation identifies rigid shapes; flexible-object masks are not guaranteed."]
                                 if self.renderer_name == "tiled" else [])}

    def routine_names(self):
        found = dict(self.namespace.get("routines") or {})
        if callable(self.namespace.get("routine")):
            found["routine"] = self.namespace["routine"]
        return sorted(found)

    def step(self, action=None, mode=""):
        self._open()
        with _scene_directory(self.path):
            self.sim.step(action, mode=mode)
        return {"step": int(self.sim.step_count), "time_s": float(self.sim.time), "finite": self.finite_report()}

    def _frame(self, *, camera=None, size=None, channels=("rgb",)):
        self._open()
        if self.renderer_name == "tiled" and "seg" in channels and self.sim.model.particle_count:
            raise ValueError("Tiled segmentation is unavailable for particle scenes in this RoboSimStudio version; capture rgb/depth and use an explicit object mask")
        spec = resolve_camera(self.sim, camera)
        if size is not None:
            if isinstance(size, str):
                size = tuple(int(part) for part in size.lower().split("x"))
            if len(size) != 2:
                raise ValueError("size must be (width,height) or WIDTHxHEIGHT")
            width, height = (_integer(v, "size", 1) for v in size)
            spec = spec.scaled(width, height)
        key = (spec, tuple(channels))
        if key not in self._renderers:
            with _scene_directory(self.path):
                renderer = self.sim.make_renderer(self.renderer_name, cameras=[spec], channels=list(channels))
                self._renderers[key] = renderer
                if self.renderer_name == "rtx":
                    renderer.render(self.sim.state)  # first call stages the RTX scene
        self.sim.sync()
        return spec, self._renderers[key].render(self.sim.state)[spec.name]

    def snapshot(self, path, *, camera=None, size=None, channels=("rgb",)):
        """Capture current state without stepping/reset; optional raw depth/seg NPZ."""
        path = Path(path).resolve()
        spec, frame = self._frame(camera=camera, size=size, channels=channels)
        path.parent.mkdir(parents=True, exist_ok=True)
        if "rgb" not in frame:
            raise ValueError("snapshot requires rgb in channels")
        Image.fromarray(np.asarray(frame["rgb"], dtype=np.uint8)).save(path)
        result = {"path": str(path), "step": int(self.sim.step_count), "time_s": float(self.sim.time),
                  "camera": spec.name, "width": spec.width, "height": spec.height,
                  "intrinsics": spec.intrinsics.tolist(), "render": self.render_report()}
        raw = {key: np.asarray(value) for key, value in frame.items() if key != "rgb"}
        if raw:
            raw_path = path.with_suffix(".npz")
            np.savez_compressed(raw_path, **raw)
            result["arrays"] = str(raw_path)
        if "seg" in raw:
            result["segmentation_names"] = self.sim.scene.handles.segmentation_names()
        return result

    def run(self, steps, *, routine=None, capture_every=0, output_dir=None, camera=None, size=None, video=False, fps=None):
        """Run up to a step budget; report early completion/errors explicitly.

        A named routine must advance exactly one control step per yield. A run
        can resume the same generator across calls. ``routine=None`` means idle
        stepping; selecting another routine closes the old one without reset.
        ``passed`` concerns execution only. ``budget_completed`` separately
        states whether the requested number of control steps actually ran.
        ``fps=None`` records at control_hz/capture_every. Capture timestamps
        are authoritative when a final capture falls between regular intervals.
        """
        self._open()
        steps = _integer(steps, "steps", 1)
        capture_every = _integer(capture_every, "capture_every")
        if (capture_every or video) and output_dir is None:
            raise ValueError("capture/video requires output_dir")
        if video and not capture_every:
            capture_every = 1
        if fps is None:
            fps = float(self.scene.physics.control_hz) / max(capture_every, 1)
        if isinstance(fps, bool) or not np.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be finite and positive")
        fps = float(fps)
        out = Path(output_dir).resolve() if output_dir is not None else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        start_step, start_time = int(self.sim.step_count), float(self.sim.time)
        report = {"status": "running", "passed": False, "requested_steps": steps,
                  "step_start": start_step, "time_start_s": start_time, "routine": routine,
                  "routine_status": "disabled" if routine is None else "running", "captures": [],
                  "render": self.render_report(), "reconstruction_validated": False}
        positions, sample_steps, sample_times = [], [], []
        state_arrays = {key: [] for key in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_qd")}
        geometry = {}
        if out:
            for name in ("particle_q", "particle_mass", "particle_inv_mass", "particle_flags", "tri_indices", "tet_indices"):
                values = getattr(self.sim.model, name, None)
                if values is not None:
                    geometry[name] = values.numpy().copy()
            for name in ("body_label", "joint_label"):
                values = getattr(self.sim.model, name, None)
                if values is not None:
                    geometry[name] = np.asarray(values, dtype=str)
            geometry_path = out / "model_geometry.npz"
            np.savez_compressed(geometry_path, **geometry)
            report["model_geometry"] = str(geometry_path)
        writer = None

        def sample(capture=False):
            if not sample_steps or sample_steps[-1] != int(self.sim.step_count):
                sample_steps.append(int(self.sim.step_count))
                sample_times.append(float(self.sim.time))
                values = self.sim.state.particle_q
                if values is not None:
                    positions.append(values.numpy().copy())
                for key, samples in state_arrays.items():
                    values = getattr(self.sim.state, key, None)
                    if values is not None:
                        samples.append(values.numpy().copy())
            if capture:
                file = out / f"frame_{int(self.sim.step_count):06d}.png"
                frame = self.snapshot(file, camera=camera, size=size)
                report["captures"].append(frame)
                if writer is not None:
                    with Image.open(file) as image:
                        writer.append_data(np.asarray(image.convert("RGB")))

        try:
            if routine != self._routine_name or (routine is not None and self._routine is None and not self._routine_finished):
                if self._routine is not None:
                    self._routine.close()
                self._routine = None
                self._routine_name = routine
                self._routine_finished = False
                if routine is not None:
                    functions = dict(self.namespace.get("routines") or {})
                    if callable(self.namespace.get("routine")):
                        functions["routine"] = self.namespace["routine"]
                    if routine not in functions or not callable(functions[routine]):
                        raise ValueError(f"Unknown routine {routine!r}; available: {self.routine_names()}")
                    with _scene_directory(self.path):
                        self._routine = functions[routine](self.sim)
                    if int(self.sim.step_count) != start_step:
                        raise RuntimeError("Routine creation must not advance physics; step inside the generator before yielding")
                    if not inspect.isgenerator(self._routine):
                        self._routine = None
                        raise TypeError("A routine must return a generator which steps the sim and yields once per control step")
            if video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(out / "rollout.mp4"), fps=fps,
                                            quality=7, macro_block_size=1)
                report["video"] = str(out / "rollout.mp4")
                report["video_fps"] = fps
                report["video_timing"] = "constant_frame_rate; exact simulation times are recorded in captures"
            sample(capture=bool(capture_every))
            if not self.finite_report()["passed"]:
                raise RuntimeError("Initial simulation state is non-finite")
            for index in range(steps):
                if routine is not None and self._routine_finished:
                    report["routine_status"] = "completed"
                    report["status"] = "completed_early"
                    break
                before = int(self.sim.step_count)
                with _scene_directory(self.path):
                    if self._routine is None:
                        self.sim.run(1)
                    else:
                        try:
                            next(self._routine)
                        except StopIteration:
                            report["routine_status"] = "completed"
                            self._routine = None
                            self._routine_finished = True
                            if int(self.sim.step_count) != before:
                                raise RuntimeError("Routine advanced physics while returning without yielding")
                            report["status"] = "completed_early"
                            break
                delta = int(self.sim.step_count) - before
                if delta != 1:
                    raise RuntimeError(f"Routine must advance exactly one control step per yield; observed {delta}")
                if not self.finite_report()["passed"]:
                    raise RuntimeError("Simulation state became non-finite")
                sample(capture=bool(capture_every and ((index + 1) % capture_every == 0 or index + 1 == steps)))
                terminated, truncated, info = self.sim.evaluate()
                if terminated or truncated:
                    report["termination"] = info
                    report["status"] = "terminated" if terminated else "truncated"
                    break
            else:
                report["status"] = "completed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            if routine is not None:
                report["routine_status"] = "error"
                if self._routine is not None:
                    self._routine.close()
                    self._routine = None
        finally:
            # A routine may finish between scheduled captures. Retain its final
            # finite state instead of ending the visible evidence at an older
            # frame. Exact timestamps distinguish this short final interval.
            if (capture_every and report["status"] != "failed" and self.finite_report()["passed"] and
                    (not report["captures"] or report["captures"][-1]["step"] != int(self.sim.step_count))):
                try:
                    sample(capture=True)
                except Exception as error:
                    report["status"] = "failed"
                    report["error"] = f"Final capture failed: {type(error).__name__}: {error}"
            if writer is not None:
                writer.close()
            report.update({"step_end": int(self.sim.step_count), "time_end_s": float(self.sim.time),
                           "completed_steps": int(self.sim.step_count) - start_step,
                           "finite": self.finite_report(), "seconds": time.perf_counter() - started})
            report["budget_completed"] = report["completed_steps"] == steps
            report["execution_passed"] = (report["status"] in {
                "completed", "completed_early", "terminated", "truncated"
            } and report["finite"]["passed"])
            report["passed"] = report["execution_passed"]
            if out and sample_steps:
                trajectory = out / "trajectory.npz"
                np.savez_compressed(trajectory, steps=np.asarray(sample_steps), times_s=np.asarray(sample_times),
                                    particle_q=np.asarray(positions),
                                    **{key: np.asarray(value) for key, value in state_arrays.items() if value})
                report["trajectory"] = str(trajectory)
                if "particle_q" in geometry:
                    report["mesh_strain"] = _mesh_strain(
                        geometry["particle_q"], geometry.get("tri_indices", np.empty((0, 3), dtype=int)),
                        geometry.get("tet_indices", np.empty((0, 4), dtype=int)), positions, sample_times)
                report["particle_ranges"] = {name: [int(i) for i in entity.particles]
                                             for name, entity in self.sim.scene.by_name.items()
                                             if getattr(entity, "particles", None) is not None}
        return report

    def close(self):
        if self.closed:
            return
        try:
            if self._routine is not None:
                self._routine.close()
            for renderer in self._renderers.values():
                renderer.close()
        finally:
            self.sim.close()
            self.closed = True

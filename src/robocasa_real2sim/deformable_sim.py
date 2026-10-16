"""Newton VBD rollouts, deformation checks and observation-based fitting.

Newton and Warp are imported only at execution time. A runtime pass concerns
the supplied loads and supports, separately from matching real observations.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np

from .deformable import (load_bundle, number, tet_volumes,
                         triangle_areas, validate_mesh, _ids)
from .io import write_json


def build_newton_model(manifest, vertices, elements, masses, young_pa=None, device="cpu"):
    """Build the same Newton shell/tet primitives used by RoboSimStudio.

    CPU is explicit, never a silent fallback from a requested CUDA device.
    """
    import newton
    import warp as wp
    version = tuple(int(v) for v in newton.__version__.split(".")[:2])
    if version != (1, 6):
        raise RuntimeError("This adapter targets Newton >=1.6,<1.7, matching RoboSimStudio")
    wp.init()
    material, simulation = manifest["material"], manifest["simulation"]
    young = material["young_pa"] if young_pa is None else young_pa
    nu = material["poisson_ratio"]
    mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    common = {"pos": wp.vec3(0, 0, 0), "rot": wp.quat_identity(), "scale": 1.0,
              "vel": wp.vec3(0, 0, 0), "vertices": [wp.vec3(*v) for v in vertices],
              "indices": elements.ravel().tolist(), "particle_radius": manifest["collision_radius_m"]}
    builder = newton.ModelBuilder(gravity=tuple(simulation["gravity_m_s2"]))
    if manifest["kind"] == "soft":
        builder.add_soft_mesh(**common, density=material["density_kg_m3"],
                              k_mu=mu, k_lambda=lam, k_damp=material["damping_s"] * mu)
    else:
        thickness = material["thickness_m"]
        # Plane-stress membrane Lame coefficients and plate bending rigidity.
        # These are an isotropic starting model, not recovered textile properties.
        mu_2d = mu * thickness
        lam_2d = young * nu * thickness / (1 - nu * nu)
        bending = young * thickness ** 3 / (12 * (1 - nu * nu))
        builder.add_cloth_mesh(**common, density=material["areal_density_kg_m2"],
                               tri_ke=mu_2d, tri_ka=lam_2d, tri_kd=material["damping_s"] * mu_2d,
                               edge_ke=bending, edge_kd=material["damping_s"] * bending)
    if builder.particle_count != len(vertices) or (manifest["kind"] == "soft" and builder.tet_count != len(elements)):
        raise RuntimeError("Newton dropped mesh vertices/elements; inspect rest topology and winding")
    for i, mass in enumerate(masses):
        builder.particle_mass[i] = 0.0 if i in manifest["pins"] else float(mass)
    if "ground_z_m" in simulation:
        builder.add_ground_plane(height=simulation["ground_z_m"])
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_mu = material["friction"]
    solver = newton.solvers.SolverVBD(
        model, iterations=simulation["iterations"],
        particle_enable_self_contact=simulation["self_contact"],
        particle_self_contact_margin=manifest["collision_radius_m"],
        particle_self_contact_gap=manifest["collision_radius_m"],
        particle_enable_tile_solve=not model.device.is_cpu,
        particle_topological_contact_filter_threshold=2,
        particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128)
    return model, solver


def nonrigid_displacement(rest, current):
    """Largest residual after the best proper rigid alignment (including bends)."""
    reference = rest - rest.mean(axis=0)
    centered = current - current.mean(axis=0)
    u, _, vt = np.linalg.svd(reference.T @ centered)
    orientation = np.eye(3)
    orientation[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ orientation @ vt
    return float(np.max(np.linalg.norm(centered - reference @ rotation, axis=1)))


def rollout(manifest, vertices, elements, masses, times, young_pa=None,
            max_edge_strain=1.0, pin_tolerance_m=1e-6, device="cpu"):
    import newton
    times = np.asarray(times, dtype=float)
    dt = manifest["simulation"]["timestep_s"]
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0 or
            np.any(np.diff(times) <= 0) or times[-1] / dt > 1_000_000):
        raise ValueError("Sample times must increase from 0 within one million simulation steps")
    steps = np.rint(times / dt).astype(np.int64)
    if np.any(np.diff(steps) <= 0) or np.max(np.abs(steps * dt - times)) > dt * 1e-5:
        raise ValueError("Sample times must start at 0, increase strictly, and lie on the simulation timestep")
    model, solver = build_newton_model(manifest, vertices, elements, masses, young_pa, device)
    state, next_state = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    _, _, _, edges, measures, _ = validate_mesh(vertices, elements, manifest["kind"])
    rest_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    pins = manifest["pins"]
    rest = state.particle_q.numpy().astype(float)
    if not np.allclose(rest, vertices, atol=1e-6, rtol=1e-6):
        raise ValueError("Compiled rest positions differ from the prepared geometry")
    frames = [rest]
    sample = 1
    metrics = {"max_edge_strain": 0.0, "max_edge_change_m": 0.0, "max_pin_error_m": 0.0,
               "minimum_element_measure_ratio": 1.0, "max_nonrigid_displacement_m": 0.0}
    max_edge_strain = number(max_edge_strain, "max_edge_strain")
    pin_tolerance_m = number(pin_tolerance_m, "pin_tolerance_m")
    # Every step is checked, even when the saved observation rate is much lower.
    for step in range(1, int(steps[-1]) + 1):
        state.clear_forces()
        forces = np.zeros((len(vertices), 3), dtype=np.float32)
        time = (step - 1) * dt
        for load in manifest["simulation"]["loads"]:
            if load["start_s"] <= time < load["end_s"]:
                np.add.at(forces, load["vertex_ids"], load["force_n"])
        state.particle_f.assign(forces)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, dt)
        state, next_state = next_state, state
        current = state.particle_q.numpy().astype(float)
        if not (np.isfinite(current).all() and np.isfinite(state.particle_qd.numpy()).all()):
            raise RuntimeError(f"Non-finite Newton dynamics at {step * dt:g} s")
        lengths = np.linalg.norm(current[edges[:, 0]] - current[edges[:, 1]], axis=1)
        strain = float(np.max(np.abs(lengths / rest_lengths - 1)))
        metrics["max_edge_strain"] = max(metrics["max_edge_strain"], strain)
        metrics["max_edge_change_m"] = max(metrics["max_edge_change_m"], float(np.max(np.abs(lengths - rest_lengths))))
        metrics["max_nonrigid_displacement_m"] = max(metrics["max_nonrigid_displacement_m"],
                                                    nonrigid_displacement(rest, current))
        if strain > max_edge_strain:
            raise RuntimeError(f"Edge strain {strain:g} exceeds {max_edge_strain:g}; refine the mesh/material/timestep")
        if pins:
            error = float(np.max(np.linalg.norm(current[pins] - rest[pins], axis=1)))
            metrics["max_pin_error_m"] = max(metrics["max_pin_error_m"], error)
            if error > pin_tolerance_m:
                raise RuntimeError("Fixed vertices moved beyond pin_tolerance_m")
        current_measure = (tet_volumes(current, elements) if manifest["kind"] == "soft"
                           else triangle_areas(current, elements))
        ratio = float(np.min(current_measure / measures))
        metrics["minimum_element_measure_ratio"] = min(metrics["minimum_element_measure_ratio"], ratio)
        if ratio <= 1e-8:
            raise RuntimeError("Inverted tetrahedron or collapsed element in deformable rollout")
        if step == steps[sample]:
            frames.append(current.copy())
            sample += 1
    return np.asarray(frames), {**metrics, "physics_steps": int(steps[-1]), "newton_version": newton.__version__, "device": str(model.device), "solver": "vbd"}


def read_observations(path, vertices, timestep_s):
    """Metric tracks with explicit correspondence; NaN triples denote occlusion."""
    with np.load(path, allow_pickle=False) as tracks:
        if any(key not in tracks for key in ("times_s", "positions_m", "vertex_ids")):
            raise ValueError("Tracks NPZ requires times_s, positions_m and vertex_ids")
        times = np.asarray(tracks["times_s"], dtype=float)
        positions = np.asarray(tracks["positions_m"], dtype=float)
        ids = _ids(tracks["vertex_ids"], len(vertices), "tracked vertex_ids")
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Tracks need at least three increasing times starting at 0")
    if np.any(np.abs(times / timestep_s - np.rint(times / timestep_s)) > 1e-5):
        raise ValueError("Observation times must align with simulation timestep_s")
    if positions.shape != (len(times), len(ids), 3):
        raise ValueError("positions_m must have shape (times, tracked vertices, 3)")
    valid = np.isfinite(positions).all(axis=-1)
    if np.isinf(positions).any() or np.any(np.isnan(positions).all(axis=-1) != ~valid):
        raise ValueError("Occluded observations must use three NaNs, not partial coordinates or infinity")
    if not valid[0].all() or np.any(np.sum(valid, axis=1) == 0):
        raise ValueError("Each frame needs a visible track and all tracked vertices must be visible at time 0")
    if not np.allclose(positions[0], vertices[ids], atol=1e-5, rtol=0):
        raise ValueError("Initial tracks must match the metric rest mesh/frame within 1e-5 m")
    return times, positions, np.asarray(ids), valid


def _rmse(predicted, observed, valid):
    return float(np.sqrt(np.mean(np.sum((predicted[valid] - observed[valid]) ** 2, axis=-1))))


def validate_deformable(bundle, output, duration_s=1.0, min_deformation_m=1e-6,
                        max_edge_strain=1.0, observations=None, max_rmse_m=0.01, device="cpu"):
    output = Path(output)
    report_path = output / "deformable_validation.json"
    write_json(report_path, {"status": "running", "passed": False})
    try:
        manifest, vertices, elements, masses = load_bundle(bundle)
        dt = manifest["simulation"]["timestep_s"]
        minimum = number(min_deformation_m, "min_deformation_m", strict=False)
        rmse_limit = number(max_rmse_m, "max_rmse_m")
        tracks = None
        if observations:
            times, tracks, ids, valid = read_observations(observations, vertices, dt)
        else:
            duration = number(duration_s, "duration_s")
            if duration / dt > 1_000_000:
                raise ValueError("Rollout exceeds one million physics steps")
            count = round(duration / dt)
            if count < 1 or abs(count * dt - duration) > dt * 1e-5:
                raise ValueError("duration_s must be a positive multiple of timestep_s")
            sample_steps = np.unique(np.rint(np.linspace(0, count, min(count + 1, 201))).astype(int))
            times = sample_steps * dt
        frames, metrics = rollout(manifest, vertices, elements, masses, times, device=device,
                                  max_edge_strain=max_edge_strain)
        np.savez_compressed(output / "trajectory.npz", times_s=times, positions_m=frames,
                            vertex_ids=np.arange(len(vertices)))
        if metrics["max_nonrigid_displacement_m"] < minimum:
            raise RuntimeError("No measurable deformation under the supplied supports/loads (rigid translation is insufficient)")
        report = {"status": "passed", "passed": True, "scope": "Newton VBD deformable dynamics under supplied supports and loads",
                  "trajectory": "trajectory.npz", "metrics": metrics,
                  "thresholds": {"min_deformation_m": minimum, "max_edge_strain": max_edge_strain,
                                 "max_rmse_m": rmse_limit if tracks is not None else None},
                  "bundle_sha256": manifest["bundle_sha256"],
                  "manifest_sha256": sha256((Path(bundle) / "manifest.json").read_bytes()).hexdigest(),
                  "real_world_calibrated": False, "robosimstudio_runtime_validated": False}
        if tracks is not None:
            rmse = _rmse(frames[:, ids], tracks, valid)
            report.update(observation_rmse_m=rmse, observation_sha256=sha256(Path(observations).read_bytes()).hexdigest(),
                          passed=rmse <= rmse_limit, status="passed" if rmse <= rmse_limit else "failed")
            if not report["passed"]:
                report["error"] = "Observed deformation mismatch exceeds max_rmse_m"
        write_json(report_path, report)
        if not report["passed"]:
            raise RuntimeError(report["error"])
        return report
    except (Exception, KeyboardInterrupt) as error:
        # Always replace previous success, including failures before compilation.
        from .io import read_json
        current = read_json(report_path)
        current.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       passed=False, error=str(error))
        write_json(report_path, current)
        raise


def fit_deformable(bundle, observations, output, young_candidates, device="cpu"):
    """Fit Young's modulus over explicit candidates with known loads and tracks.

    Poisson ratio, density, thickness and damping stay fixed. This is a fit to
    the supplied recording, not an assertion of general real-world validity.
    """
    source, output = Path(bundle).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Fit output must be separate from the input bundle")
    report_path = output / "fit.json"
    write_json(report_path, {"status": "running", "real_world_calibrated": False})
    # An old fitted model must never be represented as the result of a failed rerun.
    write_json(output / "manifest.json", {"status": "running", "dynamics_validated": False})
    write_json(output / "deformable_validation.json", {"status": "not_run", "passed": False})
    try:
        manifest, vertices, elements, masses = load_bundle(source)
        candidates = sorted({number(value, "young candidate") for value in young_candidates})
        if len(candidates) < 2 or len(candidates) > 100:
            raise ValueError("Provide 2 to 100 distinct positive Young's modulus candidates")
        times, observed, ids, valid = read_observations(observations, vertices, manifest["simulation"]["timestep_s"])
        if not set(ids) - set(manifest["pins"]):
            raise ValueError("Material fitting requires tracked free vertices")
        rows, best = [], None
        reference_prediction, sensitivity, valid_count = None, 0.0, 0
        for young in candidates:
            try:
                frames, metrics = rollout(manifest, vertices, elements, masses, times, young_pa=young, device=device)
                # Ignore the exact initial condition in the fitting objective.
                score = _rmse(frames[1:, ids], observed[1:], valid[1:])
                rows.append({"young_pa": young, "rmse_m": score, "status": "evaluated"})
                tracked_prediction = frames[:, ids][valid]
                if reference_prediction is None:
                    reference_prediction = tracked_prediction
                else:
                    sensitivity = max(sensitivity, float(np.max(np.linalg.norm(
                        tracked_prediction - reference_prediction, axis=-1))))
                valid_count += 1
                # Keep only the best complete trajectory, not one for every candidate.
                if best is None or score < best[0]:
                    best = (score, young, frames, metrics)
            except (RuntimeError, ValueError) as error:
                rows.append({"young_pa": young, "status": "failed", "error": str(error)})
        if valid_count < 2:
            raise RuntimeError("Fewer than two valid candidate rollouts; refine material range/timestep")
        if sensitivity < 1e-7:
            raise ValueError("Observed vertices are insensitive to Young's modulus; add deformation, supports or known loads")
        score, young, frames, metrics = best
        fitted = deepcopy(manifest)
        fitted["material"]["young_pa"] = young
        fitted["material"]["provenance"]["young_pa"] = "fitted"
        fitted.update(dynamics_validated=False, real_world_calibrated=False)
        for name in ("geometry.npz", "surface.obj"):
            shutil.copy2(source / name, output / name)
        report = {"status": "fitted", "parameter": "young_pa", "best_young_pa": young, "rmse_m": score,
                  "candidate_results": rows, "sensitivity_m": sensitivity,
                  "at_search_boundary": young in (candidates[0], candidates[-1]),
                  "observation_sha256": sha256(Path(observations).read_bytes()).hexdigest(),
                  "input_bundle_sha256": manifest["bundle_sha256"], "metrics": metrics,
                  "input_manifest_sha256": sha256((source / "manifest.json").read_bytes()).hexdigest(),
                  "validation_scope": "training recording only; verify on independent observations",
                  "real_world_calibrated": False}
        fitted["fit"] = "fit.json"
        np.savez_compressed(output / "fit_trajectory.npz", times_s=times, positions_m=frames,
                            vertex_ids=np.arange(len(vertices)))
        write_json(output / "manifest.json", fitted)
        write_json(report_path, report)
        return report
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(report_path, {"status": status, "error": str(error), "real_world_calibrated": False})
        write_json(output / "manifest.json", {"status": status, "error": str(error), "dynamics_validated": False})
        raise

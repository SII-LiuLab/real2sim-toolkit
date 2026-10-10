"""State-isolated COLMAP reconstruction with logs and explicit scale status."""

from pathlib import Path
import os
import shutil
import struct
import subprocess
import time
import tempfile
import uuid
from hashlib import sha256

from PIL import Image, ImageOps

from .io import read_json, write_json


def find_colmap(explicit=None):
    candidate = explicit or os.environ.get("REAL2SIM_COLMAP") or shutil.which("colmap")
    if candidate:
        candidate = shutil.which(str(candidate)) or candidate
        candidate = Path(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(f"COLMAP executable missing: {candidate}")
        return str(candidate.resolve())
    return None


def reconstruction_plan(capture_report, state, part="assembly"):
    group = [item for item in capture_report["images"] if item.get("accepted") and item["state"] == state and item.get("part", "assembly") == part]
    return {"state": state, "part": part, "images": group,
            "status": "ready_for_sfm" if len(group) >= 3 else "capture_required",
            "reason": None if len(group) >= 3 else "At least three unique overlapping views of one unchanged state/part are required.",
            "metric_scale_known": False,
            "articulation": "Part segmentation and joint fitting are separate from static surface reconstruction."}


def command_plan(executable, work, has_masks=False):
    work = Path(work)
    db, images = str(work / "database.db"), str(work / "images")
    extraction = [executable, "feature_extractor", "--database_path", db, "--image_path", images,
                  "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "0",
                  "--SiftExtraction.max_image_size", "2000"]
    if has_masks:
        extraction += ["--ImageReader.mask_path", str(work / "masks")]
    return [extraction,
            [executable, "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "0"],
            [executable, "mapper", "--database_path", db, "--image_path", images, "--output_path", str(work / "sparse")]]


def stage_inputs(plan, work, previous=None):
    """Resume only when image order, pixels and foreground masks are unchanged."""
    masks_present = any(item.get("mask") for item in plan["images"])
    if previous:
        if bool(previous.get("masked")) != masks_present:
            raise ValueError("Foreground masks changed; start a new reconstruction")
        if len(list((work / "images").glob("*.png"))) != len(plan["images"]):
            raise ValueError("Input photo count changed; start a new reconstruction")
    sizes, fingerprints = set(), []
    for i, item in enumerate(plan["images"]):
        name = f"{i:05d}.png"
        with Image.open(item["path"]) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
        sizes.add(oriented.size)
        image_destination = work / "images" / name
        if previous:
            with Image.open(image_destination) as staged:
                if staged.size != oriented.size or staged.convert("RGB").tobytes() != oriented.tobytes():
                    raise ValueError("Input photos changed; start a new reconstruction")
        else:
            oriented.save(image_destination)
        fingerprint = {"name": name, "size": list(oriented.size),
                       "pixel_sha256": sha256(oriented.tobytes()).hexdigest()}
        if masks_present:
            if item.get("mask"):
                with Image.open(item["mask"]) as source:
                    mask = source.convert("L").point(lambda x: 255 if x > 0 else 0)
            else:
                mask = Image.new("L", oriented.size, 255)
            if mask.size != oriented.size:
                raise ValueError("Mask dimensions differ from the EXIF-oriented image")
            destination = work / "masks" / f"{name}.png"
            if previous:
                with Image.open(destination) as staged:
                    if staged.size != mask.size or staged.convert("L").tobytes() != mask.tobytes():
                        raise ValueError("Foreground masks changed; start a new reconstruction")
            else:
                mask.save(destination)
            fingerprint["mask_sha256"] = sha256(mask.tobytes()).hexdigest()
        fingerprints.append(fingerprint)
    if len(sizes) != 1:
        raise ValueError("Shared-camera reconstruction requires images with the same dimensions and lens setting")
    return masks_present, fingerprints


def ply_counts(path):
    """Read counts without loading a potentially large binary mesh into RAM."""
    counts = {}
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        for _ in range(1000):
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                return counts
            if line.startswith("element "):
                _, kind, count = line.split()
                counts[kind] = int(count)
    raise ValueError(f"Missing PLY header terminator: {path}")


def check_dense_geometry(cloud, surface):
    points = ply_counts(cloud).get("vertex", 0)
    mesh = ply_counts(surface)
    if points < 3:
        raise RuntimeError("Dense fusion produced too few points; inspect overlap, masks and depth maps")
    if mesh.get("vertex", 0) < 3 or mesh.get("face", 0) < 1:
        raise RuntimeError("Meshing produced an empty surface; review point-cloud coverage and meshing parameters")
    return {"fused_points": points, "mesh_vertices": mesh["vertex"], "mesh_faces": mesh["face"],
            "review_required": True, "watertight_checked": False}


def run_reconstruction(plan, output, executable=None, dense=False, jobs_root=None, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if plan["status"] != "ready_for_sfm":
        if resume:
            raise ValueError("Resume requires the original set of accepted input photos")
        write_json(output / "reconstruction.json", plan)
        return plan
    executable = find_colmap(executable)
    if not executable:
        raise FileNotFoundError("Install COLMAP or provide --colmap / REAL2SIM_COLMAP")
    # Isolate native Windows paths from Unicode user directories.
    jobs_root = Path(jobs_root or os.environ.get("REAL2SIM_JOBS") or Path(tempfile.gettempdir()) / "real2sim-jobs")
    if os.name == "nt" and not str(jobs_root).isascii() and not resume:
        raise ValueError("COLMAP on Windows needs an ASCII work path; specify --jobs-root or REAL2SIM_JOBS")
    report_path = output / "reconstruction.json"
    previous = read_json(report_path) if resume and report_path.is_file() else None
    if resume and (not previous or not previous.get("workdir")):
        raise ValueError("No saved reconstruction workspace to resume")
    if previous and (previous.get("state"), previous.get("part")) != (plan["state"], plan["part"]):
        raise ValueError("Cannot resume a different state or moving part")
    work = Path(previous["workdir"]) if previous else jobs_root / uuid.uuid4().hex
    if previous and not work.is_dir():
        raise ValueError("Saved reconstruction workspace is missing")
    for folder in ("images", "masks", "sparse", "dense", "logs"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    masks_present, fingerprints = stage_inputs(plan, work, previous)
    result = {key: value for key, value in plan.items() if key != "images"}
    commands = previous.get("commands", []) if previous else []
    completed = previous.get("completed_stages", [item["argv"][1] for item in commands[:-1]]) if previous else []
    result.update(status="running", workdir=str(work), commands=commands,
                  completed_stages=completed, masked=masks_present, inputs=fingerprints)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")

    def run(command):
        stage = command[1]
        if previous and stage in result["completed_stages"]:
            saved = next((item for item in reversed(commands) if item["argv"][1] == stage), None)
            if saved and saved["argv"] == command:
                print(f"COLMAP: {stage} (saved result)", flush=True)
                return
            if stage != "poisson_mesher":
                raise ValueError(f"Saved {stage} settings changed; start a new reconstruction")
            result["completed_stages"].remove(stage)
        logfile = work / "logs" / f"{len(result['commands']):02d}-{stage}.log"
        result["commands"].append({"argv": command, "log": str(logfile)})
        write_json(report_path, result)
        print(f"COLMAP: {stage}", flush=True)
        with logfile.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            try:
                while process.poll() is None:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        result["commands"][-1]["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"COLMAP {stage} failed; see {logfile}")
        result["completed_stages"].append(stage)
        write_json(report_path, result)

    try:
        for command in command_plan(executable, work, masks_present):
            run(command)
        models = list((work / "sparse").glob("*/images.bin"))
        if not models:
            raise RuntimeError("No camera model recovered; add overlapping, sharp, textured views")
        def image_count(path):
            with path.open("rb") as stream:
                return struct.unpack("<Q", stream.read(8))[0]
        selected = max(models, key=image_count)
        result["registered_images"] = image_count(selected)
        if result["registered_images"] < 3:
            raise RuntimeError("Too few registered cameras for a usable multiview reconstruction")
        model = selected.parent
        run([executable, "model_converter", "--input_path", str(model), "--output_path", str(model), "--output_type", "TXT"])
        shutil.copytree(model, output / "sparse", dirs_exist_ok=True)
        if dense:
            dense_root = work / "dense"
            run([executable, "image_undistorter", "--image_path", str(work / "images"), "--input_path", str(model), "--output_path", str(dense_root), "--output_type", "COLMAP", "--max_image_size", "1600"])
            run([executable, "patch_match_stereo", "--workspace_path", str(dense_root), "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", "1600", "--PatchMatchStereo.cache_size", "2"])
            run([executable, "stereo_fusion", "--workspace_path", str(dense_root), "--input_type", "geometric", "--output_path", str(dense_root / "fused.ply")])
            points = ply_counts(dense_root / "fused.ply").get("vertex", 0)
            if points < 3:
                raise RuntimeError("Dense fusion produced too few points; inspect overlap, masks and depth maps")
            # COLMAP's default trim=10 can delete the whole surface on small captures.
            # Keep a conservative surface for review; do not infer watertight collision geometry.
            run([executable, "poisson_mesher", "--input_path", str(dense_root / "fused.ply"), "--output_path", str(dense_root / "surface.ply"), "--PoissonMeshing.trim", "5"])
            geometry = check_dense_geometry(dense_root / "fused.ply", dense_root / "surface.ply")
            for name in ("fused.ply", "surface.ply"):
                shutil.copy2(dense_root / name, output / name)
            result["surface_mesh"] = "surface.ply"
            result["geometry"] = {**geometry, "poisson_trim": 5}
        result.update(status="surface_reconstructed" if dense else "sparse_reconstructed",
                      next_steps=["metric scale and axis alignment", "moving-part segmentation and joint fitting", "texture baking and collision geometry", "MJCF export and physical validation"])
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted; resume with unchanged inputs and --resume")
        write_json(report_path, result)
        raise
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(report_path, result)
        raise
    write_json(report_path, result)
    return result

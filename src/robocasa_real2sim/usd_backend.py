"""Portable mesh export and PhysX conveyor configuration; no simulator imports."""

from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np

from .io import write_json


def _usd():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as error:
        raise RuntimeError("USD tools require pip install '.[usd]' in the toolkit environment") from error
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vec3(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    return array


def _copy_file(source, output):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing asset dependency: {source}")
    digest = sha256(source.read_bytes()).hexdigest()
    relative = Path("assets") / f"{digest[:20]}{source.suffix.lower()}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return relative.as_posix()


def _mesh(stage, path, source, scale):
    from pxr import Gf, UsdGeom
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("Mesh export requires the usd extra (trimesh)") from error
    mesh = trimesh.load(str(source), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float) * scale
    faces = np.asarray(mesh.faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.size or not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has no finite vertices: {source}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
        raise ValueError(f"Mesh has no triangle faces: {source}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"Mesh has invalid triangle indices: {source}")
    usd = UsdGeom.Mesh.Define(stage, path)
    usd.CreatePointsAttr(vertices.tolist())
    usd.CreateFaceVertexCountsAttr([3] * len(faces))
    usd.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    usd.CreateSubdivisionSchemeAttr("none")
    usd.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None and len(uv) == len(vertices):
        from pxr import Sdf
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
            raise ValueError(f"Mesh has invalid UV coordinates: {source}")
        UsdGeom.PrimvarsAPI(usd).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv.tolist())
    colors = getattr(mesh.visual, "vertex_colors", None)
    if getattr(mesh.visual, "kind", None) == "vertex" and colors is not None:
        usd.CreateDisplayColorPrimvar("vertex").Set((np.asarray(colors)[:, :3] / 255).tolist())
    return usd


def _material(stage, mesh, path, appearance, output):
    from pxr import Gf, Sdf, UsdShade
    color = _vec3(appearance.get("color_rgb", [0.65, 0.65, 0.65]), "color_rgb")
    if np.any(color < 0) or np.any(color > 1):
        raise ValueError("color_rgb must be in [0, 1]")
    roughness = float(appearance.get("roughness", 0.6))
    if not np.isfinite(roughness) or not 0 <= roughness <= 1:
        raise ValueError("roughness must be in [0, 1]")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if "color_rgb" not in appearance and mesh.GetDisplayColorPrimvar().HasValue():
        reader = UsdShade.Shader.Define(stage, path + "/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    texture = appearance.get("texture")
    if texture:
        from pxr import UsdGeom
        if not UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st"):
            raise ValueError("Texture needs mesh UV coordinates; texture baking is not automatic")
        reader = UsdShade.Shader.Define(stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        sampler = UsdShade.Shader.Define(stage, path + "/Texture")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(_copy_file(texture, output)))
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _surface_velocity(prim, speed, direction, units):
    from pxr import Gf, Sdf, UsdPhysics
    direction = _vec3(direction, "conveyor direction")
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Conveyor direction must be nonzero")
    speed = float(speed)
    if not np.isfinite(speed):
        raise ValueError("Conveyor speed must be finite (negative reverses direction)")
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("The belt prim must already have CollisionAPI")
    ancestor = prim.GetParent()
    while ancestor and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Belt cannot be nested under another rigid body; separate the collision surface first")
        ancestor = ancestor.GetParent()
    # Token + typed attributes work with usd-core. Isaac supplies the registered schema.
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    prim.AddAppliedSchema("PhysxSurfaceVelocityAPI")
    prim.CreateAttribute("physxSurfaceVelocity:surfaceVelocityEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("physxSurfaceVelocity:surfaceVelocityLocalSpace", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("physxSurfaceVelocity:surfaceVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(*(direction / norm * speed / units)))
    prim.CreateAttribute("physxSurfaceVelocity:surfaceAngularVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0))


def _build_mesh_usd(request, output):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, _ = _usd()
    output = Path(output)
    parts = request.get("geometry", {}).get("parts")
    if not parts:
        raise ValueError("geometry.parts must list the visual mesh and physics for each part")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    rows, names = [], set()
    for part in parts:
        name = part["name"]
        if not Sdf.Path.IsValidIdentifier(name) or name in names or name == "Materials":
            raise ValueError("Part names must be distinct USD identifiers, excluding Materials")
        names.add(name)
        if part.get("aligned_z_up") is not True:
            raise ValueError(f"Align {name} into a Z-up frame before export; set aligned_z_up=true")
        scale = _positive(part.get("scale_m_per_unit", 0), "scale_m_per_unit")
        path = "/Asset/" + name
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*_vec3(part.get("position_m", [0, 0, 0]), "position_m")))
        visual = _mesh(stage, path + "/Visual", part["mesh"], scale)
        _material(stage, visual, "/Asset/Materials/" + name, part.get("appearance", {}), output)
        kind = part.get("physics", "static")
        if kind not in {"static", "rigid", "belt"}:
            raise ValueError("Mesh export supports static, rigid, belt; use MJCF for articulated parts")
        approximation = part.get("collision_approximation", "none" if kind == "static" else "convexHull")
        if approximation not in {"none", "convexHull", "convexDecomposition"}:
            raise ValueError("Unsupported collision_approximation")
        if kind != "static" and approximation == "none":
            raise ValueError("Moving meshes need convexHull or convexDecomposition collision")
        if part.get("requires_open_interior") and approximation == "convexHull":
            raise ValueError("convexHull closes the interior of cups/bowls; supply convex pieces or convexDecomposition")
        if kind == "rigid":
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(_positive(part.get("mass_kg", 0), "mass_kg"))
        # Explicit collision meshes share the visual mesh's frame and scale.
        collisions = part.get("collision_meshes", [])
        geometries = []
        if collisions:
            for i, source in enumerate(collisions):
                collision = _mesh(stage, path + f"/Collision_{i}", source, scale)
                collision.CreateVisibilityAttr("invisible")
                geometries.append(collision)
        else:
            geometries = [visual]
        for geometry in geometries:
            UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr(approximation)
            if kind == "belt":
                if len(geometries) != 1:
                    raise ValueError("A belt part needs one collision prim; configure each belt separately")
                belt = request.get("conveyor", {})
                if "speed_m_s" not in belt:
                    raise ValueError("conveyor.speed_m_s is required for belt physics")
                _surface_velocity(geometry.GetPrim(), belt["speed_m_s"], belt.get("direction_local", [1, 0, 0]), 1)
        rows.append({"name": name, "body_prim": path, "physics": kind,
                     "collision_prims": [str(geom.GetPath()) for geom in geometries],
                     "scale_m_per_unit": scale, "scale_source": part.get("scale_source", "assumed"),
                     "mass_source": part.get("mass_source", "assumed") if kind == "rigid" else None,
                     "source_mesh_sha256": sha256(Path(part["mesh"]).read_bytes()).hexdigest(),
                     "appearance_source": "configured texture/color or vertex colors"})
    output.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output / "asset.usda")):
        raise RuntimeError("USD export failed")
    manifest = {"status": "prepared", "model": "asset.usda", "object_id": request["id"],
                "category": request["category"], "units": "m", "up_axis": "Z", "parts": rows,
                "isaac_runtime_validated": False, "real_world_calibrated": False,
                "limitations": ["No automatic articulation, scale recovery, texture baking, or physical parameter fitting"]}
    write_json(output / "manifest.json", manifest)
    return manifest


def build_mesh_usd(request, output):
    report = Path(output) / "manifest.json"
    write_json(report, {"status": "running", "isaac_runtime_validated": False})
    try:
        return _build_mesh_usd(request, output)
    except (Exception, KeyboardInterrupt) as error:
        write_json(report, {"status": "failed", "error": str(error), "isaac_runtime_validated": False})
        raise


def prepare_usd(source, output, belt_prim=None, speed=None, direction=(1, 0, 0)):
    """Flatten a local USD and bundle referenced material files before portability."""
    Gf, Sdf, Usd, UsdGeom, _, _ = _usd()
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source.parent or output in source.parents:
        raise ValueError("Use an output folder separate from the source USD and its dependencies")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "manifest.json"
    write_json(report_path, {"status": "running", "isaac_runtime_validated": False})
    try:
        original = Usd.Stage.Open(str(source))
        if not original:
            raise ValueError(f"Could not open USD: {source}")
        if original.GetCompositionErrors():
            raise ValueError(f"USD has unresolved composition dependencies: {original.GetCompositionErrors()}")
        if not original.HasAuthoredMetadata("metersPerUnit") or not original.HasAuthoredMetadata("upAxis"):
            raise ValueError("Source USD must explicitly declare metersPerUnit and upAxis")
        stage = Usd.Stage.Open(original.Flatten())
        units = _positive(UsdGeom.GetStageMetersPerUnit(stage), "metersPerUnit")
        # Flatten resolves composed asset values. Make their files relative again.
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() not in {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray}:
                    continue
                if attr.GetTimeSamples():
                    raise ValueError(f"Animated asset paths require explicit packaging: {attr.GetPath()}")
                value = attr.Get()
                if value is None:
                    continue
                def local_asset(asset):
                    if not asset.path:
                        return asset
                    resolved = asset.resolvedPath or asset.path
                    if not Path(resolved).is_absolute():
                        resolved = source.parent / resolved
                    return Sdf.AssetPath(_copy_file(resolved, output))
                attr.Set(local_asset(value) if attr.GetTypeName() == Sdf.ValueTypeNames.Asset else [local_asset(asset) for asset in value])
        if belt_prim:
            prim = stage.GetPrimAtPath(belt_prim)
            if not prim:
                raise ValueError(f"Belt prim not found: {belt_prim}")
            _surface_velocity(prim, speed, direction, units)
        if not stage.GetRootLayer().Export(str(output / "asset.usda")):
            raise RuntimeError("USD export failed")
        manifest = {"status": "prepared", "model": "asset.usda", "meters_per_unit": units,
                    "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "belt_prim": belt_prim,
                    "speed_m_s": speed, "isaac_runtime_validated": False, "real_world_calibrated": False}
        write_json(report_path, manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        write_json(report_path, {"status": "failed", "error": str(error), "isaac_runtime_validated": False})
        raise


def inspect_usd(source):
    _, _, Usd, UsdGeom, UsdPhysics, _ = _usd()
    stage = Usd.Stage.Open(str(Path(source).resolve()))
    if not stage:
        raise ValueError(f"Could not open USD: {source}")
    return {"meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "units_authored": stage.HasAuthoredMetadata("metersPerUnit"),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "rigid_bodies": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)],
            "colliders": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)],
            "joints": [{"path": str(p.GetPath()), "type": p.GetTypeName()} for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)],
            "belts": [str(p.GetPath()) for p in stage.Traverse() if p.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Get()],
            "isaac_runtime_validated": False}

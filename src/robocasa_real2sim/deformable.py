"""Metric rest meshes and material descriptions for deformable reconstruction.

Like RoboSimStudio's ClothSpec/SoftSpec, shells and volumes are different
physical models. No rigid convex hull or unconstrained Delaunay fill is used.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np

from .io import read_json, write_json


SOURCES = {"measured", "retrieved", "assumed", "fitted"}


def number(value, name, minimum=0, strict=True):
    value = float(value)
    if not np.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}")
    return value


def vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return value


def indices(value, width, count, name):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != width or not len(value):
        raise ValueError(f"{name} must have shape (n, {width}) with n > 0")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if value.min() < 0 or value.max() >= count:
        raise ValueError(f"{name} indices are outside the vertex array")
    value = value.astype(np.int64)
    ordered = np.sort(value, axis=1)
    if np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError(f"{name} contains repeated vertices in an element")
    if len(np.unique(ordered, axis=0)) != len(value):
        raise ValueError(f"{name} contains duplicate elements")
    if len(np.unique(value)) != count:
        raise ValueError("Remove unused vertices before assigning pins or tracking vertex IDs")
    return value


def tet_volumes(vertices, tets):
    p = vertices[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6


def triangle_areas(vertices, faces):
    p = vertices[faces]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2


def surface_topology(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ordered = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts > 2):
        raise ValueError("Non-manifold surface edge; repair the source mesh")
    if np.any((counts == 2) & (orientation != 0)):
        raise ValueError("Inconsistent surface winding; orient the source triangles first")
    return unique, counts


def check_connected(elements, count):
    parent = np.arange(count)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for element in elements:
        for i in element[1:]:
            parent[root(int(i))] = root(int(element[0]))
    if len({root(i) for i in range(count)}) != 1:
        raise ValueError("Use one connected deformable per bundle; separate disconnected components")


def validate_mesh(vertices, elements, kind):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must have shape (n, 3) and finite coordinates")
    elements = indices(elements, 3 if kind == "cloth" else 4, len(vertices), "faces" if kind == "cloth" else "tets")
    check_connected(elements, len(vertices))
    p = vertices[elements]
    size = np.max(np.linalg.norm(p[:, :, None] - p[:, None, :], axis=-1), axis=(1, 2))
    if kind == "cloth":
        measures = triangle_areas(vertices, elements)
        if np.any(measures <= size ** 2 * 1e-10):
            raise ValueError("Degenerate or very thin triangles; remesh the rest surface")
        edges, counts = surface_topology(elements)
        faces = elements
        details = {"area_m2": float(measures.sum()), "boundary_edges": int(np.sum(counts == 1))}
    else:
        volumes = tet_volumes(vertices, elements)
        if np.any(np.abs(volumes) <= size ** 3 * 1e-10):
            raise ValueError("Degenerate or very thin tetrahedra; remesh the rest volume")
        # Newton discards negatively wound tets. Preserve vertex IDs and fix winding.
        elements = elements.copy()
        flipped = volumes < 0
        elements[flipped, :2] = elements[flipped, 1::-1]
        oriented = np.concatenate([elements[:, face] for face in ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1])])
        _, inverse, counts = np.unique(np.sort(oriented, axis=1), axis=0, return_inverse=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Non-manifold tetrahedral face")
        # Adjacent positive tets must lie on opposite sides of their shared face.
        parity = np.where(((oriented[:, 0] > oriented[:, 1]).astype(int) +
                           (oriented[:, 0] > oriented[:, 2]).astype(int) +
                           (oriented[:, 1] > oriented[:, 2]).astype(int)) % 2, -1, 1)
        balance = np.bincount(inverse, weights=parity)
        if np.any((counts == 2) & (balance != 0)):
            raise ValueError("Tetrahedra overlap across a shared face")
        faces = oriented[counts[inverse] == 1]
        surface_topology(faces)
        edges = np.unique(np.sort(np.concatenate([elements[:, pair] for pair in
                          ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])]), axis=1), axis=0)
        measures = np.abs(volumes)
        details = {"volume_m3": float(measures.sum()), "reoriented_tets": int(flipped.sum())}
    return vertices, elements, faces, edges, measures, details


def _read_mesh(path, kind, tetrahedralization):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as mesh:
            if "vertices" not in mesh:
                raise ValueError("NPZ needs vertices")
            key = "faces" if kind == "cloth" else "tets"
            if key not in mesh:
                raise ValueError(f"NPZ for {kind} needs {key}; tetrahedralize a surface mesh explicitly")
            return mesh["vertices"], mesh[key], "provided_rest_mesh"
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if kind == "cloth":
        return vertices, faces, "provided_rest_surface"
    if not tetrahedralization:
        raise ValueError("Soft bodies require NPZ vertices/tets or explicit tetrahedralize=true for a closed surface")
    vertices, faces, _, _, _, _ = validate_mesh(vertices, faces, "cloth")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Tetrahedralization requires a watertight, consistently wound surface; no automatic hole filling")
    try:
        import tetgen
    except ImportError as error:
        raise ImportError("Surface-to-volume meshing needs the tetra extra: pip install -e '.[tetra]'") from error
    # Constrained PLC meshing preserves concavities; convex Delaunay filling does not.
    generator = tetgen.TetGen(vertices, faces)
    result = generator.tetrahedralize(order=1, quality=True, nobisect=True, minratio=1.5)
    if len(result[0]) < len(vertices) or not np.allclose(result[0][:len(vertices)], vertices, rtol=0, atol=1e-10):
        raise ValueError("Tetrahedralizer changed source vertex ordering; export a tetrahedral NPZ and reassign pins/tracks")
    return np.asarray(result[0]), np.asarray(result[1]), "tetgen_constrained_surface"


def normalize_material(spec, kind):
    material = dict(spec)
    provenance = material.pop("provenance", {})
    required = ["young_pa", "poisson_ratio", "areal_density_kg_m2", "thickness_m"] if kind == "cloth" else ["young_pa", "poisson_ratio", "density_kg_m3"]
    allowed = set(required) | {"damping_s", "friction"}
    if set(material) - allowed:
        raise ValueError(f"Unknown material parameters: {sorted(set(material) - allowed)}")
    if any(key not in material for key in required):
        raise ValueError(f"{kind} material requires {', '.join(required)}")
    material.setdefault("damping_s", 0.005)
    material.setdefault("friction", 0.5)
    for key in material:
        material[key] = number(material[key], key, strict=key not in {"poisson_ratio", "damping_s", "friction"})
    if material["poisson_ratio"] >= 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    material["provenance"] = {key: provenance.get(key, "assumed") for key in material}
    if any(source not in SOURCES for source in material["provenance"].values()):
        raise ValueError(f"Material provenance must be one of {sorted(SOURCES)}")
    return material


def _ids(value, count, name, allow_empty=False):
    raw = np.asarray(value)
    if allow_empty and raw.shape == (0,):
        return []
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or len(raw) == 0:
        raise ValueError(f"{name} must be a list of integer vertex IDs")
    if raw.min() < 0 or raw.max() >= count or len(np.unique(raw)) != len(raw):
        raise ValueError(f"{name} contains duplicate or out-of-range vertex IDs")
    return raw.astype(int).tolist()


def _simulation(spec, count, pins):
    simulation = dict(spec)
    if set(simulation) - {"timestep_s", "gravity_m_s2", "ground_z_m", "loads", "iterations", "self_contact"}:
        raise ValueError("Unknown simulation field; supported: timestep_s, gravity_m_s2, ground_z_m, loads, iterations, self_contact")
    iterations = simulation.get("iterations", 10)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 200:
        raise ValueError("iterations must be an integer in [1, 200]")
    simulation["iterations"] = iterations
    contact = simulation.get("self_contact", True)
    if not isinstance(contact, bool):
        raise ValueError("self_contact must be true or false")
    simulation["self_contact"] = contact
    simulation["timestep_s"] = number(simulation.get("timestep_s", 0.001), "timestep_s")
    simulation["gravity_m_s2"] = vector(simulation.get("gravity_m_s2", [0, 0, -9.81]), "gravity_m_s2").tolist()
    if "ground_z_m" in simulation:
        z = float(simulation["ground_z_m"])
        if not np.isfinite(z):
            raise ValueError("ground_z_m must be finite")
        simulation["ground_z_m"] = z
    loads = []
    for load in simulation.get("loads", []):
        ids = _ids(load["vertex_ids"], count, "load vertex_ids")
        if set(ids) & set(pins):
            raise ValueError("Loads must act on free vertices, not fixed pins")
        start = number(load.get("start_s", 0), "load start_s", strict=False)
        end = number(load["end_s"], "load end_s")
        if end <= start:
            raise ValueError("load end_s must exceed start_s")
        loads.append({"vertex_ids": ids, "force_n": vector(load["force_n"], "force_n").tolist(),
                      "start_s": start, "end_s": end})
    simulation["loads"] = loads
    return simulation


def _values(array):
    return " ".join(format(float(x), ".17g") for x in np.asarray(array).ravel())


def build_deformable(request, output):
    output = Path(output)
    report_path = output / "manifest.json"
    write_json(report_path, {"status": "running", "dynamics_validated": False})
    write_json(output / "deformable_validation.json", {"status": "not_run", "passed": False})
    try:
        spec = request.get("deformable", {})
        kind = spec.get("kind")
        if kind not in {"cloth", "soft"}:
            raise ValueError("deformable.kind must be cloth (triangles) or soft (tetrahedra)")
        if request.get("simulator", "newton") not in {"newton", "robosimstudio"}:
            raise ValueError("Deformable export uses Newton; request simulator=newton or robosimstudio. No Isaac adapter is implemented")
        if spec.get("aligned_z_up") is not True:
            raise ValueError("Align the rest mesh into Z-up and set aligned_z_up=true")
        if not spec.get("rest_state") or spec.get("rest_source") not in SOURCES:
            raise ValueError("Declare rest_state and rest_source; a loaded/deformed scan is not automatically stress-free")
        if spec.get("scale_source") not in SOURCES:
            raise ValueError("scale_source is required (measured/retrieved/assumed/fitted)")
        scale = number(spec.get("scale_m_per_unit", 0), "scale_m_per_unit")
        position = vector(spec.get("position_m", [0, 0, 0]), "position_m")
        source = Path(spec["mesh"]).resolve()
        if source == (output / "geometry.npz").resolve() or source == (output / "surface.obj").resolve():
            raise ValueError("Use an output folder separate from the input rest mesh")
        vertices, elements, method = _read_mesh(source, kind, spec.get("tetrahedralize") is True)
        vertices = np.asarray(vertices, dtype=float) * scale + position
        vertices, elements, faces, edges, measures, details = validate_mesh(vertices, elements, kind)
        material = normalize_material(spec.get("material", {}), kind)
        density = material["areal_density_kg_m2" if kind == "cloth" else "density_kg_m3"]
        masses = np.zeros(len(vertices))
        for column in elements.T:
            np.add.at(masses, column, measures * density / elements.shape[1])
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("Mesh scale/density produced invalid nodal masses")
        pins = _ids(spec.get("pins", []), len(vertices), "pins", allow_empty=True)
        if len(pins) == len(vertices):
            raise ValueError("At least one vertex must remain free to deform")
        radius = number(spec.get("collision_radius_m", material.get("thickness_m", .002) / 2), "collision_radius_m")
        simulation = _simulation(spec.get("simulation", {}), len(vertices), pins)
        manifest = {"schema_version": 1, "status": "prepared", "object_id": request["id"],
                    "category": request["category"], "behavior": "deformable", "kind": kind,
                    "simulator": request.get("simulator", "newton"), "units": "m", "up_axis": "Z",
                    "backend": "newton", "solver": "vbd", "newton_version_profile": ">=1.6,<1.7",
                    "geometry": "geometry.npz", "surface": "surface.obj",
                    "material": material, "pins": pins, "collision_radius_m": radius,
                    "simulation": simulation, "rest_state": spec["rest_state"], "rest_source": spec["rest_source"],
                    "scale_m_per_unit": scale, "scale_source": spec["scale_source"], "position_m": position.tolist(),
                    "geometry_method": method, "source_mesh_sha256": sha256(source.read_bytes()).hexdigest(),
                    "vertex_count": len(vertices), "element_count": len(elements),
                    "mass_kg": float(masses.sum()), "geometry_checks": details,
                    "dynamics_validated": False, "robosimstudio_runtime_validated": False,
                    "real_world_calibrated": False,
                    "limitations": ["Rest geometry, metric frame and material observations must be supplied",
                                    "Isotropic elastic material; no plasticity, tearing or automatic video tracking",
                                    "Geometry checks do not detect all self-intersections or overlapping tetrahedra",
                                    "No automatic texture recovery or Isaac deformable conversion"]}
        np.savez_compressed(output / "geometry.npz", vertices=vertices, faces=faces,
                            **({"tets": elements} if kind == "soft" else {}), masses_kg=masses)
        with (output / "surface.obj").open("w", encoding="utf-8") as stream:
            for row in vertices:
                stream.write("v " + _values(row) + "\n")
            for face in faces:
                stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")
        manifest["bundle_sha256"] = {name: sha256((output / name).read_bytes()).hexdigest()
                                     for name in ("geometry.npz", "surface.obj")}
        write_json(report_path, manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        write_json(report_path, {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                 "error": str(error), "dynamics_validated": False})
        raise


def load_bundle(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("status") != "prepared" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a prepared deformable bundle")
    for name in ("geometry.npz", "surface.obj"):
        if sha256((folder / name).read_bytes()).hexdigest() != manifest["bundle_sha256"].get(name):
            raise ValueError(f"Bundle file changed: {name}; rebuild before validation or fitting")
    with np.load(folder / "geometry.npz", allow_pickle=False) as mesh:
        vertices = mesh["vertices"]
        elements = mesh["faces" if manifest["kind"] == "cloth" else "tets"]
        masses = mesh["masses_kg"]
    return manifest, vertices, elements, masses

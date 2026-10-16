# Flexible reconstruction with Newton

This reference covers the portable `build-deformable`, `validate-deformable` and `fit-deformable` commands, which run Newton VBD directly. Use `simulator: "newton"` and `behavior: "deformable"`; the compatibility destination `simulator: "robosimstudio"` is also accepted. For an editable native UI scene, use real2sim's [session workflow](sessions.md). Its internal native runtime is in `src/robosimstudio*`, not a separate installation. The direct bundle commands alone do not build that UI scene or validate its interaction. Isaac deformable conversion is unsupported and requests targeting it fail explicitly.

## Native editable scenes

Use `rss.sheet`, `rss.sponge` and `rss.rod` for native cloth, soft-volume and rod scenes. Their implementations live in `src/robosimstudio/_domain_packs/cloth.py`, `soft.py` and `rod/`; compatibility imports remain `robosimstudio.domains.cloth`, `.soft` and `.rod`. Read the relevant implementation when authoring unfamiliar material, contact or support behavior. Case geometry and routines belong in the session. Native UI/capture uses the [session contract](sessions.md); the portable bundle contract below is a separate direct-Newton path.

## From capture to a flexible asset

1. Capture a stationary reference configuration. For a stress-free model, unload the object; a sagged/hanging scan can contain pre-strain. Declare `rest_state` and `rest_source`, including an assumption when the reference cannot be unloaded. Inspect photos, then run `reconstruct --state <reference> --dense` with the existing COLMAP path, or supply an external scan.
2. Segment and remesh the object into simulation geometry. Cloth needs one triangle **mid-surface**, not two disconnected scan skins or a closed towel-shaped volume. A soft solid needs a closed outer surface for constrained tetrahedralization, or supplied tetrahedra. Choose a resolution suited to bending/contact and review it; this tool does not automatically extract a cloth mid-surface, decimate scans, repair holes or recover a stress-free shape.
3. Align into Z-up, set measured/assumed metric scale, and author material, supports and known forces. Build with `build-deformable` or `pipeline`. Inspect the portable geometry and parameter provenance in `manifest.json`.
4. Run `validate-deformable --device cpu` or `--device cuda:0`. The same Newton VBD solver runs on both devices; CUDA requires a visible compatible NVIDIA driver. Devices never silently fall back. Runtime imports remain optional for geometry preparation and CLI help.
5. When available, provide metric vertex tracks from a known deformation experiment. Fit Young's modulus over explicit candidates with `fit-deformable`, then validate against an **independent** recording using `validate-deformable --observations ... --max-rmse ...`. Fitting does not infer force, density, Poisson ratio, friction, thickness or damping.

## Request contract

The shared `id` and `category` remain required. Input paths resolve relative to the request JSON. `deformable` contains:

| Field | Contract |
|---|---|
| `kind` | `cloth` for triangle shell or `soft` for volumetric tetrahedra |
| `mesh` | NPZ (`vertices` and `faces` or `tets`) or trimesh-readable triangle mesh |
| `tetrahedralize` | Explicit `true` to convert a watertight soft-body surface with the optional `tetra` extra; never a convex fill |
| `scale_m_per_unit`, `scale_source` | Positive explicit scale and `measured`/`retrieved`/`assumed`/`fitted` provenance |
| `aligned_z_up` | Must be `true`; alignment is not inferred |
| `position_m` | Translation after scaling, default `[0,0,0]`; no automatic recentering |
| `rest_state`, `rest_source` | Nonempty state label and the same provenance vocabulary |
| `pins` | Unique zero-based source vertex IDs, fixed at their prepared positions; default `[]` |
| `collision_radius_m` | Positive Newton particle/self-contact radius; defaults to half cloth thickness or 1 mm for solids |
| `material` | Required material quantities below; unknown keys fail rather than being ignored |
| `simulation` | Timestep, gravity, solver iterations, self-contact, optional known ground and loads |

NPZ arrays: `vertices` is finite `(N,3)`; `faces` is integer `(M,3)`; `tets` is integer `(M,4)`. Vertex order is preserved. Unused vertices, repeated/invalid indices, degenerate elements, disconnected components, non-manifold surface edges and inconsistent triangle winding are rejected. Negative tetrahedral winding is corrected without renumbering vertices. Meshes still need geometric review: these checks do not establish absence of all self-intersections or nonadjacent overlapping tetrahedra. Constrained TetGen meshing preserves a single connected outer boundary; use a supplied tetrahedral NPZ for cavities, multiple materials or carefully controlled interior topology. TetGen can add vertices; inspect generated `geometry.npz` before assigning pins/tracks that refer to added nodes.

Material quantities (all SI):

| Quantity | Cloth | Soft solid |
|---|---|---|
| `young_pa` | Positive isotropic elastic modulus | Positive Young's modulus |
| `poisson_ratio` | `[0,0.5)` | `[0,0.5)` |
| `areal_density_kg_m2` | Positive and required | Not used |
| `thickness_m` | Positive and required | Not used |
| `density_kg_m3` | Not used | Positive and required |
| `damping_s` | Nonnegative, default `0.005` | Nonnegative, default `0.005` |
| `friction` | Nonnegative, default `0.5` | Nonnegative, default `0.5` |
| `provenance` | Map quantity names to measured/retrieved/assumed/fitted | Same |

Missing provenance is recorded as `assumed`. Solids convert Young/Poisson to 3D Lamé coefficients. Cloth uses a plane-stress isotropic membrane and bending rigidity `E*h^3/(12*(1-nu^2))`; real woven cloth is often anisotropic and requires a richer calibrated constitutive model. Rest dihedral angles come from the source surface. Nodal masses are lumped from triangle areas or tet volumes; fixed nodes have zero inverse mass in Newton.

`simulation` fields:

| Field | Meaning |
|---|---|
| `timestep_s` | Positive substep duration, default `0.001` s |
| `gravity_m_s2` | Finite vector, default `[0,0,-9.81]` |
| `iterations` | Integer VBD iterations per substep in `[1,200]`, default `10` |
| `self_contact` | Boolean, default `true`; Newton surface self-contact |
| `ground_z_m` | Optional measured/authored Z of a stationary support plane; absent means no plane |
| `loads` | Array of `vertex_ids`, `force_n`, `start_s` (default 0), required `end_s` |

`force_n` is a world-frame force **per selected vertex**, active on `[start_s,end_s)`. Overlapping loads add. Loads on fixed pins fail. These are known experimental boundary conditions, not inferred robot contacts. No robot, gripper or demonstration scene is created.

## Outputs and validation

`build-deformable` writes `manifest.json`, metric `geometry.npz` (including lumped `masses_kg`) and `surface.obj`. Original scans are not needed after packaging. SHA-256 hashes detect changed geometry before execution. `manifest.json` retains solver configuration and source/scale/material provenance; status remains `prepared` and `dynamics_validated` remains false. `deformable_validation.json` is the authoritative runtime report for the specific experiment, not a permanent claim in the asset manifest.

Validation records `trajectory.npz`, evaluates every physics substep for finite state, excessive edge strain, pin drift, tet inversion or collapsed triangles, and requires measurable nonrigid displacement after the best proper rigid alignment (default 1 micrometre). Pure translation or rotation is insufficient; cloth bending can pass even when local edge lengths barely change. `--min-deformation 0` permits a stability-only experiment. Default maximum absolute edge strain is `1.0`; refine it for the material. Contact geometry/materials, mesh resolution and timestep still need experiment-specific review. The checker does not certify collision-free folding, frictional grasping or robot contact accuracy.

All reruns replace prior reports with running/failed/interrupted status before work. A failed build or fit invalidates the output manifest. A fit writes a separate output bundle and cannot overwrite the input.

## Observations and material fitting

Provide an NPZ with:

- `times_s`: finite increasing `(T,)`, at least three samples, starting at zero, aligned to `timestep_s`.
- `vertex_ids`: distinct integer `(K,)` IDs in the prepared simulation mesh.
- `positions_m`: `(T,K,3)` observations in the same metric Z-up frame. The first sample matches the reference vertices within 10 micrometres. A fully occluded vertex uses three NaNs; partial NaNs/infinity fail. Every frame needs at least one visible track.

Tracks must describe the same topology throughout the experiment. Independently reconstructed meshes need explicit correspondence before use. Automatic 2D/3D tracking, camera optimization and nonrigid registration are not implemented.

`fit-deformable --young-pa ...` evaluates 2–100 distinct positive candidate moduli. It uses position RMSE over visible observations after the initial frame, excludes failed rollouts, and rejects recordings insensitive to Young's modulus (for example unconstrained rigid free motion). `fit.json` records every candidate, the best modulus, training RMSE, parameter sensitivity and a warning flag when the optimum lies at a search boundary. The fitted modulus provenance becomes `fitted`; `real_world_calibrated` stays false. Observation validation is a separate step with an explicit `--max-rmse` threshold.

## Newton scene integration

The public helper builds a Newton model/solver without adding a robot or viewer:

```python
from robocasa_real2sim.deformable import load_bundle
from robocasa_real2sim.deformable_sim import build_newton_model

manifest, vertices, elements, masses = load_bundle("outputs/flexible")
model, solver = build_newton_model(manifest, vertices, elements, masses, device="cuda:0")
```

`build_newton_model` uses `ModelBuilder.add_cloth_mesh` / `add_soft_mesh`, preserves pins, colors bending constraints, and creates `SolverVBD`. The runtime uses Newton's collision pipeline and explicit external particle forces. This adapter targets `newton>=1.6,<1.7`, coordinated with real2sim's internal native runtime in `src/robosimstudio`. Keep source/license records under `third_party/robosimstudio` and record local edits in Git. The external Newton/RTX dependency lock remains separate from editable source. Keep Newton API migrations and robot coupling as distinct target-runtime checks.

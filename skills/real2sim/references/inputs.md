# Input contracts

Paths in a request JSON resolve relative to that JSON. Keep captures/datasets outside Git. Labels are explicit; category and motion type are not automatically inferred from photos.

## Shared request

| Field | Contract |
|---|---|
| `id`, `category` | Nonempty identifier and semantic category |
| `behavior` | Intended `static`, `rigid`, `articulated`, `conveyor`; checked when candidates declare behavior |
| `joint_types` | Movable `hinge`/`slide`/`ball`; fixed joints/free base are not mechanism joints |
| `captures` | Array of `path`, explicit `state`, optional `part` and `mask` |
| `geometry.size_m` | Native fixture dimensions `[width, depth, height]` in metres |
| `geometry.dimension_source` | `measured`/`assumed`; dimensions alone do not calibrate dynamics |
| `geometry.panel_type` | Native `slab`, `shaker`, `raised` |
| `appearance.reference` | `path`, optional normalized `bbox: [left, top, right, bottom]` of one surface |
| `appearance.apply_best_texture` | Apply local wood texture heuristic; native `geometry.texture` takes priority |

Native geometry also accepts `thickness_m`, `door_gap_m`, `levels`, `handle_type`, `handle_length_m`, and single-door `orientation`. `fixture.yaml` contains native parameters; MJCF pivot fixes do not automatically propagate into a RoboCasa room YAML.

## Capture practice

Capture whole-object and overlapping side/oblique views with consistent exposure where practical. Capture interiors, handles, edges and visible hinges separately. Aim for roughly 60–80% neighboring view overlap; count depends on shape/occlusion. Include a known length in the same rigid scene and record endpoints for scale recovery. Metric scaling/alignment currently require manual work.

Capture closed, intermediate and open mechanisms as separate groups. Hold each component at one unchanged state during its multiview capture. Record travel/angle and limits with measurement. Record belt speed independently; still photos cannot determine speed.

Masks match EXIF-corrected dimensions: 0 background, 255 foreground. Keep state and objects unchanged within a static group. Reflective/transparent/textureless surfaces may need retrieved or simplified geometry. Blur checks are advisory. Do not merge moving states into one COLMAP model.

## Local metadata

Each indexed folder has `asset.json` with `id`, `category`, relative `model` and `license`. Optional `source_url`, `behavior` and geometry metadata aid review. MJCF includes must be expanded; URDF movable joints are parsed. USD needs explicit `joints` (including `[]` for static/belt assets); declared USD topology still needs verification. Missing behavior metadata means unverified behavior. URDF indexing is supported; automatic URDF conversion is not implemented.

## Metric mesh export

`build-usd` uses `geometry.parts`, an array of independent component descriptions:

| Part field | Contract |
|---|---|
| `name` | Distinct USD identifier, excluding `Materials` |
| `mesh` | Triangle mesh in a trimesh-supported format |
| `scale_m_per_unit`, `scale_source` | Positive explicit scale and provenance; never assume SfM metres |
| `aligned_z_up` | `true` after aligning components into a common Z-up frame |
| `position_m` | Translation in the asset frame; default `[0,0,0]` |
| `physics` | `static`, `rigid`, `belt`; articulation needs another authored format |
| `mass_kg`, `mass_source` | Positive required mass for `rigid`, with `measured`/`assumed` provenance |
| `collision_meshes` | Optional collision files in the visual mesh's frame/scale |
| `collision_approximation` | `none` for static triangles, `convexHull`/`convexDecomposition` for moving parts |
| `requires_open_interior` | Reject convex hull closure; decomposition still needs simulator inspection |
| `appearance` | `color_rgb` and `roughness` in `[0,1]`; optional `texture` with existing UVs |

For belt physics, request-level `conveyor.speed_m_s` is required. `conveyor.direction_local` defaults to `[1,0,0]`. Each belt part has one collision surface. Visual and collision frames must agree. Texture baking and optical material recovery are not implemented; configured colors/textures are appearance approximations.

All parts export in metres with Z up. Hidden joints, masses and friction are not inferred. Supply authored contact/material properties through an existing USD for experiments needing calibrated friction.

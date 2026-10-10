---
name: real2sim
description: Turn real photos, scans or existing assets into simulation assets for household objects, furniture and industrial equipment. Use for asset screening, multiview reconstruction, MJCF/USD authoring and RoboCasa or Isaac Sim experiments.
---

# Real2Sim

Use compatible existing assets first, reconstruct missing geometry when necessary, and author the behavior required by the task. Photographic similarity alone does not establish an interactive asset.

Locate the checkout from the working directory or `REAL2SIM_REPO`. Use its installed `real2sim` CLI (or `python -m robocasa_real2sim`); run `doctor` to establish available providers. When this skill is installed outside the checkout, scripts remain in the repository, not inside the installed skill folder.

## Route by interaction

- Static furniture and structures: metric visual mesh and suitable static collision.
- Graspable objects: rigid body, mass and collision; preserve openings when containment matters.
- Doors, drawers, windows and other mechanisms: separate moving links, authored axes, limits and clearance. Prefer verified articulated assets. Unknown joints need observation or authoring; label guessed parameters.
- Conveyors and driven equipment: identify the actual driven surface/link and controls. Texture animation cannot substitute for object transport. The current backend covers straight contact transport on an authored kinematic surface; other mechanisms need behavior adapters.

Read [inputs.md](references/inputs.md) when preparing captures, requests, dataset metadata or mesh exports. Read [isaac.md](references/isaac.md) for Isaac import, behavior configuration and verification.

## Execute and retain evidence

1. Establish category, required interaction, simulator and measurements. Continue asset/geometry work while unknown mass/friction/speed are recorded as assumptions; obtain measurements when calibration requires them.
2. Inspect captures and group by unchanged state and part. Screen local assets by explicit category/topology/behavior. If online search is needed and available, research authoritative source pages, check usable geometry and mechanisms, and record source/license metadata before indexing downloaded assets. The CLI has no online/image-embedding search.
3. Review compatible candidates for scale, handles, interiors and materials. Reuse suitable assets. Otherwise reconstruct static groups with COLMAP and inspect nonempty geometry. Correct scale/frame, separate parts, and author collisions/mechanics before export.
4. Export MJCF or USD with portable dependencies. Run `isaac-prepare` before MJCF import to preserve compiled inertia. Use `build-usd` for aligned metric meshes or `prepare-usd` for existing USD. Respect cup/bowl interiors and belt contact geometry.
5. Verify the requested behavior in the target simulator. Separate compilation, structural inspection and actual physical motion. Keep reports/trajectories and investigate failed checks.

Label measured, retrieved, assumed and fitted parameters. Mark an asset `prepared` until runtime checks execute. A synthetic/tooling test is not reconstruction of the user's photos; stable default physics is not real-world calibration.

Produce the requested asset and concise run instructions. Include unfinished behavior/calibration steps that affect the experiment. This workflow needs no demonstration scene or example asset.

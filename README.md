# Real2Sim Toolkit

Asset-first real2sim for household objects, furniture, and industrial equipment. The reusable skill coordinates capture inspection, asset screening, reconstruction, physics authoring, and simulator checks. The Python CLI executes repeatable steps and writes reports. Supply your own photographs, meshes, and simulation assets.

## Linux setup

```bash
git clone real2sim-toolkit.bundle real2sim-toolkit
cd real2sim-toolkit
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,usd]'
real2sim doctor
```

Install RoboCasa separately using its [official installation instructions](https://robocasa.ai/docs/build/html/introduction/installation.html) for cabinet/drawer templates. Other operations do not require RoboCasa. Install COLMAP separately for photo reconstruction. The reconstruction CLI was exercised with COLMAP 3.11.1; newer CLI option changes may require adaptation. Use `--colmap`, `REAL2SIM_COLMAP`, or an executable on `PATH`. Use `--jobs-root` or `REAL2SIM_JOBS` for persistent reconstruction work; otherwise the system temporary folder is used. Preserve that folder for `--resume`.

Isaac scripts run with the simulator's `python.sh`, separately from `.venv`. They import the local toolkit source automatically. Do not install `usd-core` into Isaac's environment: Isaac already supplies USD and PhysX bindings.

## Skill and agent

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/real2sim "${CODEX_HOME:-$HOME/.codex}/skills/"
export REAL2SIM_REPO="$(pwd)"
```

Invoke `$real2sim` with your input folder, required interaction, and target simulator. Start the agent in this checkout, or provide `REAL2SIM_REPO`. Another agent can use the same `SKILL.md`; `AGENTS.md` describes the codebase. A separate agent server is not required.

## Pipeline

`observations → capture inspection → compatible asset screening → reuse or reconstruction → scale / parts / physics → MJCF or USD → simulator verification`

| Operation | Entry point | Result |
|---|---|---|
| Inspect photos | `real2sim inspect` | Decode, EXIF, exact duplicate and blur checks; state/part grouping |
| Screen available assets | `real2sim select --assets` | Category, joint topology and behavior filters; rejection reasons |
| Adapt native fixtures | `real2sim pipeline` | RoboCasa cabinets/drawers; portable MJCF and MuJoCo checks |
| Reconstruct static surfaces | `real2sim reconstruct --state --dense` | COLMAP cameras, cloud and nonempty mesh |
| Export aligned metric meshes | `real2sim build-usd` | Static objects, rigid bodies and belt surfaces; UV texture/color |
| Package existing USD | `real2sim prepare-usd` | Flatten local composition and bundle file attributes; optional belt behavior |
| Prepare MJCF | `real2sim isaac-prepare` | Bundle files and freeze compiled body mass/COM/inertia |
| Inspect USD | `real2sim usd-inspect` | Units, colliders, bodies, joints and belt settings |
| Import in Isaac | `scripts/isaac_import.py` | MJCF-to-USD and movable joint topology check |
| Check contact transport | `scripts/isaac_conveyor.py` | Existing body's forward/reverse/stop physics trajectory |

Use `--help` for arguments. Read [input contracts](skills/real2sim/references/inputs.md) and [Isaac execution](skills/real2sim/references/isaac.md) for request fields and runtime steps.

## Current boundaries

The local index accepts MJCF, URDF and USD with semantic/license metadata. Remote indexing/downloads, image embeddings and automatic photo recognition are not implemented. The skill can research external sources when available; the CLI does not search an online dataset. Structural scores are not visual similarity scores. Wood texture ranking is a local color/contrast/grain-direction heuristic and needs surface-only crops.

Reconstruction needs one physical state per part/assembly. Three distinct views are the startup minimum, not a capture guarantee. `--resume` checks image order, pixels and masks. Dense MVS needs a CUDA-capable COLMAP build. Static scan output still needs metric scale, alignment, part separation, joint authoring, texture baking and collision geometry before interactive export. Those steps are not automatically inferred.

`build-usd` requires explicit scale and Z-up alignment per part, and mass for dynamic bodies. Open receptacles require suitable collision pieces or decomposition: a convex hull closes cups/bowls. Articulated mechanisms use authored MJCF/URDF/USD; this exporter does not infer joints. Conveyors use a kinematic collision surface with PhysX surface velocity. Flexible belts, roller mechanics, throughput and robot control need separate behavior work.

Isaac runtime adapters target the **5.x PhysX API profile**, using the official 5.1 documentation. They stop on another major version. Isaac 6.x changes the importer/backend and needs a dedicated adapter. This boundary is not an installation recommendation.

Import success verifies USD and movable joint type counts. It does not establish stable joint motion, correct materials, contact behavior or calibrated dynamics. The conveyor runner requires your existing belt and dynamic body already placed on it; it creates no demonstration scene. It records physics poses and fails when transport checks fail.

Run `python -m pytest` for CPU tests. Native integration skips without RoboCasa. Tests cover portability, scale, inertia, textures, state isolation, resumed inputs and failure reports. **Isaac GPU/runtime execution remains to be verified on your Linux machine.** A `prepared` report is not a simulation pass.

Sources: [RoboCasa fixtures](https://robocasa.ai/docs/build/html/assets/fixtures.html), [COLMAP 3.11](https://colmap.github.io/legacy/3.11/tutorial.html), [MJCF](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html), [Isaac MJCF import](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_mjcf.html), [conveyor utility](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html), [6.0 importer changes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html).

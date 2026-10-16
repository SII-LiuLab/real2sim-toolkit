# Real2Sim Toolkit

An agent harness for turning real observations into editable simulation assets.
It combines reconstruction tools, persistent sessions, video reference and replay,
Newton flexible dynamics, and a browser UI for inspecting and dragging assets.
The coding agent inspects, authors, compares and revises through the `real2sim` CLI.

Selected RoboSimStudio scene and UI code is adapted inside this distribution.
There is one repository, one installable project and one user-facing CLI.
Upstream tests, demos, benchmarks, training tools and robot model collections
are excluded. See [source attribution](THIRD_PARTY_NOTICES.md).

```text
src/robocasa_real2sim/            # Harness, reconstruction, sessions, context, evidence
src/robosimstudio/               # Internal scene and Newton runtime; flexible domains
src/robosimstudio_studio/        # Internal browser UI, picking, dragging and recording
src/robosimstudio_real2sim_agent/ # Internal live controller and video comparison
skills/real2sim/                 # Coding-agent workflow
requirements/                   # Runtime profiles
third_party/robosimstudio/       # Licenses and source provenance
scripts/                        # Isaac import and contact adapters
```

Internal import names are retained for compatibility with authored scenes.
They do not require a separate RoboSimStudio installation or checkout.

## Install

Use CPython 3.12 for the interactive Newton runtime. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install uv
uv pip install -e . -r requirements/studio.txt --override requirements/studio-overrides.txt
real2sim doctor
real2sim runtime-status
```

For RTX rendering, use `requirements/studio-rtx.txt` instead and check
`real2sim runtime-status --renderer rtx`. This downloads a large NVIDIA wheel.
The profiles pin Newton, Warp and related interfaces; runtime metadata readiness
does not establish device or physics success.

Lightweight asset and capture tools need only `pip install -e .`.
Optional exports and geometry processing use `.[usd]`, `.[deformable]` or
`.[tetra]`. Install COLMAP separately for static photo reconstruction; use
`--colmap`, `REAL2SIM_COLMAP` or an executable on `PATH`. RoboCasa templates
require a separate RoboCasa installation.

## Use with a coding agent

Read [skills/real2sim/SKILL.md](skills/real2sim/SKILL.md), or install that directory
as your coding agent's `real2sim` skill. Start the agent in this checkout or set
`REAL2SIM_REPO` to its absolute path. The toolkit supplies tools and persistent
state; the coding assistant supplies the agent.

```bash
real2sim agent --describe
real2sim session-init --output /path/to/session --reference /path/to/video.mp4
real2sim ref-frames --session /path/to/session /path/to/video.mp4 --times 0 3 6 --max-frames 3
# Author /path/to/session/scene.py, then open its UI:
real2sim agent --session /path/to/session --device cuda:0
```

The UI opens paused and reconnects to an existing page when available. Its
compact drive panel provides asset dragging, two grab handles, release and reset.
Use `--resume` to start physics immediately or `--panels all` for additional
controls. `agent` serves an authored scene; it does not embed an LLM or infer a
complete scene from the reference video.

Bind a native `rss.Scene` to `scene` in `scene.py` using `import robosimstudio as rss`.
Flexible constructors include `rss.sheet`, `rss.sponge` and `rss.rod`; Newton VBD
handles their flexible dynamics. Put meshes, textures and any robot assets in
the session and resolve them relative to `__file__`. No robot model collection
is preinstalled. Start with the target asset and necessary support; add robots
and environment when the requested interaction needs them.

See [session authoring and UI controls](skills/real2sim/references/sessions.md)
and [flexible geometry contracts](skills/real2sim/references/deformable.md).

## Video evidence and iteration

Use `scene-build` to compile the authored session, `scene-run --record` to
record its motion, and `scene-compare` to compare rendered and reference images.
Add `--compact` for bounded results. A routine accepts `sim`, advances one
control step and yields per step. Saved trajectories can be re-rendered with
`scene-render` after appearance changes without repeating physics. Reports
retain timestamps, source hashes, geometry, states and runtime provenance.
`session-ctl` operates a running page; `session-status` and `session-log` inspect
recorded results.

Use RTX for appearance comparisons; tiled rendering previews colliders.
Video references, extracted frames, editable geometry, playback and comparison
are supported. Automatic recovery of camera calibration, rest geometry,
tracking and physical parameters from arbitrary videos is not implemented.
Finite execution, visual agreement and calibrated mechanics require different
evidence. Preserve failed attempts and label authored motion explicitly.

## Persistent agent context

```bash
real2sim session-brief --session /path/to/session
real2sim session-focus --session /path/to/session --mode asset --goal 'Show the draggable asset' --validation interaction
real2sim memory-recall --session /path/to/session --query Newton --limit 5
```

The bounded brief restores scope, prioritized memories, recent evidence and UI
state. Save explicit public decisions with `memory-put` and milestones with
`session-checkpoint`, using the current focus revision. Scope or evidence changes
invalidate stale next actions. Workflow metadata lives in the session's `.agent/`
directory and is excluded from its scene fingerprint. Follow the
[control loop](skills/real2sim/references/control-loop.md) for memory, retries,
scope changes and deciding what validation is needed.

## Other asset workflows

| Task | CLI |
|---|---|
| Inspect captures and screen reusable assets | `inspect`, `select` |
| Adapt cabinet and drawer templates | `pipeline` |
| Reconstruct an unchanged static surface with COLMAP | `reconstruct` |
| Prepare cloth or tetrahedral soft geometry | `build-deformable` |
| Run Newton dynamics or fit supplied metric tracks | `validate-deformable`, `fit-deformable` |
| Prepare aligned metric USD or portable MJCF | `build-usd`, `prepare-usd`, `isaac-prepare` |
| Inspect USD units, bodies and joints | `usd-inspect` |

Use `--help` and the [input contracts](skills/real2sim/references/inputs.md).
Static scans still need scale, alignment, parts, materials and collision authoring.
Moving frames must not be fused into one static surface. Flexible fitting needs
known supports, loads and supplied metric tracks.

Isaac adapters in `scripts/` run through the simulator's own `python.sh` and
target the 5.x PhysX API profile. Keep its USD environment separate from this
venv. See [Isaac workflow](skills/real2sim/references/isaac.md) for supported
import and contact checks. Interactive Newton assets do not imply flexible
Isaac export support.

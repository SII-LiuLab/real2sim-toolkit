---
name: real2sim
description: Reconstruct or refine editable simulation assets from real photos, videos, scans or existing geometry with the real2sim harness. Use for asset inspection, native scene UI, Newton flexible dynamics and supported export workflows.
---

# Real2Sim

Real2sim is the single installable harness with integrated native scene/UI modules in `src/robosimstudio*`; those compatibility imports are internal runtime code, not a separate RoboSimStudio project. Newton VBD handles flexible dynamics. The coding agent observes, authors and revises through these tools. Keep module responsibilities, source/license provenance and pinned Newton/RTX dependencies. The CLI does not embed an LLM or automatically recover arbitrary video geometry, materials and contact.

Locate the checkout from the working directory or `REAL2SIM_REPO` and use its installed `real2sim` CLI. For an existing session, first restore `session-brief --session <session>` and reconcile it with the latest user request. Follow the [control loop](references/control-loop.md) for scope changes, memory, check selection and bounded retries. Reuse relevant evidence and a working live page. Prefer bounded briefs and `--compact` operation results; read full reports only to resolve a specific gap.

Match the deliverable to the current request. Asset mode prioritizes an inspectable, draggable target with necessary support and camera. Add a robot, complete environment or action sequence when requested; robot models are optional session assets, not an assumed bundled catalog. On a switch from a full scene to an asset, stop unrelated environment/material research and motion experiments, preserve that work, and expose the asset UI promptly. A temporary restriction such as “no tests for this change” applies to its stated scope, not all future work.

Use compatible existing assets first; preserve openings, moving parts and the required interaction. Record units, frames, scale/rest-state assumptions and measured/retrieved/fitted parameters. Sparse guides and prescribed robot motion must be labeled as authored mechanics.

Read only the reference needed for the next action:

- [agent.md](references/agent.md): ownership, runtime boundaries, UI entry and success levels.
- [sessions.md](references/sessions.md): native scene/routine contract, manipulation UI, capture/compare, portable dependencies and evidence.
- [deformable.md](references/deformable.md): Newton cloth triangle shells, soft-solid tetrahedra, rest states, loads and portable bundle/fitting commands.
- [inputs.md](references/inputs.md): references, source/license metadata, screening and geometry preparation.
- [isaac.md](references/isaac.md): explicitly requested Isaac import, supported export and runtime checks. Preserve the requested target and identify unsupported flexible conversion.

For iterative reconstruction: inspect → author native scene → preview/manipulate → validate the requested level → revise or deliver. Create sessions with `session-init`; open/reuse their UI with `agent --session`. Use RTX for appearance review and `scene-render` for saved trajectories after appearance edits, without repeating physics. Narrow inspection, tooling and supported export tasks use only the relevant stages.

Deliver the requested asset, working UI or tool change with portable dependencies and concise use instructions. Link relevant evidence and distinguish preparation, UI interaction, execution, visual agreement, dynamics checks and calibration. A draggable cloth, finite run or high silhouette IoU supports only its demonstrated claim. Export USD/MJCF only when requested and supported; preserve failed attempts and unresolved behavior affecting the result.

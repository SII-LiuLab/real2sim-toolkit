# Real2Sim Toolkit

Read [skills/real2sim/SKILL.md](skills/real2sim/SKILL.md) for real2sim work. Follow the current user scope: an asset inspection task needs the target, support and manipulation controls, not a complete robot scene.

Real2sim is the single installable product. `src/robocasa_real2sim` owns inputs, sessions, execution policy and evidence. Selected native scene, UI, dragging and rendering modules retain the internal compatibility names `src/robosimstudio`, `src/robosimstudio_studio` and `src/robosimstudio_real2sim_agent`; they are not separate projects or installation targets. Newton VBD supplies flexible dynamics. Modify the module that owns the behavior. Preserve [upstream provenance](third_party/robosimstudio/UPSTREAM.json), [LICENSE](third_party/robosimstudio/LICENSE), [NOTICE](third_party/robosimstudio/NOTICE) and asset-specific licenses.

Restore ongoing work with `real2sim session-brief --session PATH`. Reconcile new user steering before dependent work; keep temporary constraints in their focus and stop unrelated candidates when scope narrows. Use bounded memory/context and `--compact` results; read full evidence for a specific gap. See [control-loop.md](skills/real2sim/references/control-loop.md). The reserved `.agent/` directory stores workflow metadata, never simulation assets.

Create editable sessions with `session-init`; `real2sim agent --session PATH` opens or reuses their native UI. It does not create/reset a scene or embed an LLM. Keep one writer per session and reuse its live page. Use captured trajectories with `scene-render` for appearance changes instead of repeating physics. See [sessions.md](skills/real2sim/references/sessions.md).

Keep runtime imports lazy so CPU tooling works without simulators. Check changed/unknown dependencies with `runtime-status --renderer tiled|rtx`; Newton/RTX pins are separate from editable workspace source. Installation inputs are `requirements/studio.txt`, `requirements/studio-rtx.txt` and `requirements/studio-overrides.txt`. Metadata readiness does not establish GPU or simulation success.

Preserve units, frames, vertex identity, rest state, parameter provenance and portable dependencies. Distinguish preparation, interaction, execution, visual agreement, dynamics and calibration. Retain failed attempts and exact source for selected results. No arbitrary-video reconstruction or calibrated-mechanics claim without evidence. Do not add demonstrations, example assets or unrelated tooling unless requested.

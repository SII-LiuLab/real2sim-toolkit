# Agent contract

Real2sim is one installable product that owns reconstruction sessions, native UI and evidence. Selected scene/UI modules absorbed from RoboSimStudio retain `robosimstudio*` import names for compatibility; they are internal runtime code, not a separate project or install. An external coding agent uses the tools to inspect inputs, write native scenes and evaluate revisions. `real2sim agent` is the session/UI entrypoint, not an embedded LLM, a model API client or an automatic video-to-twin service.

| Component | Responsibility |
| --- | --- |
| Real2sim harness | Preserve inputs, manage editable sessions and attempts, enforce runtime policy, capture/compare/export evidence. |
| Coding agent | Observe the reference, choose geometry and assumptions, author the asset and requested mechanics, inspect results and revise. |
| Internal native runtime, `src/robosimstudio*` | Compile native scenes and provide the local UI, manipulation controls, execution and rendering within the real2sim installation. |
| Newton VBD | Simulate flexible cloth, rods and soft volumes. Articulated domains can retain their appropriate native backend. |

## Entry and scope

Start from the current request and restore an existing session with `session-brief --session <session>`. Follow the [control loop](control-loop.md) to reconcile memory, scope changes and the next useful action. Reuse relevant evidence rather than restarting the entire pipeline after each continuation.

When runtime compatibility is unknown or changed, use `real2sim runtime-status --renderer rtx` or `--renderer tiled` to inspect pinned third-party dependencies and confirm that native imports resolve to this real2sim installation. Readiness covers metadata/source resolution, not GPU/driver or simulator execution. RTX has additional dependencies; a missing RTX package does not by itself rule out tiled execution. Local harness/runtime edits are workspace changes, not automatic dependency-lock failures. Record actual third-party version changes separately.

Keep per-asset work in sessions and harness policy in `src/robocasa_real2sim`. Native scene/physics fixes belong in `src/robosimstudio`, browser controls in `src/robosimstudio_studio`, and live loading/RPC in `src/robosimstudio_real2sim_agent`. Preserve the [upstream record](../../../third_party/robosimstudio/UPSTREAM.json), [LICENSE](../../../third_party/robosimstudio/LICENSE), [NOTICE](../../../third_party/robosimstudio/NOTICE), [citation](../../../third_party/robosimstudio/CITATION.cff) and asset-specific licenses.

For a new reconstruction session, use `session-init --output <session> --reference <input>`, inspect its registered references, and author `scene.py` before launching it. Reuse an existing suitable session. Open its UI with:

```sh
real2sim agent --session <session> --renderer rtx
```

An existing live page is reused and its current URL/status returned. Without a server, the command serves the page in the foreground. It does not create, replace or reset the scene. `--panels` defaults to `drive`; `--resume` explicitly starts stepping. An already running simulation remains in its current state when its page is reused. Device/port/renderer options belong to launching the runtime; inspect the returned state rather than assuming a reused server changed configuration.

Choose the smallest scene that satisfies the requested result. Asset mode needs the target asset, necessary support/collision, useful views and native manipulation controls. Cloth should be inspectable and draggable with the runtime's cloth panel/handles. A complete robot and reconstructed surroundings belong to a requested scene or interaction task, not every asset task. Read [sessions.md](sessions.md) for the native scene and interaction contract.

## Operate toward the current deliverable

Preserve input provenance, author the native asset, expose its UI, then validate only the requested level. Resume stepping when dynamic manipulation is requested; a paused viewport alone does not demonstrate interaction. The [control loop](control-loop.md) governs revisions and retries. On explicit scope reductions, stop unrelated work promptly and retain its checkpoint; completing a previous full-scene plan must not delay an asset-only UI.

Use observations to guide authoring, not to imply recovered mechanics. Sparse prescribed grips, force guides or recorded robot motion are useful when labeled. They do not prove recovered friction, contact or material parameters. Do not substitute full-vertex animation for flexible simulation and describe it as physical reconstruction.

## Report success at the demonstrated level

| Claim | Required evidence |
| --- | --- |
| Prepared asset | Nonempty geometry and explicit units, frames, rest state and dependencies. |
| Built scene | Successful runtime compilation and initial state checks in the stated environment. |
| Interactive asset | The native UI displays the asset and the requested manipulation controls work; describe whether physics was stepping. |
| Completed execution | Actual steps or normal routine completion, recorded timestamps and runtime diagnostics. |
| Reviewed visual agreement | Reference/render pairs at relevant states, camera assumptions and explicit mask semantics where metrics are used. |
| Checked dynamics | Relevant contact/deformation diagnostics, such as true mesh-edge strain, intersections, drift and release behavior, under the stated loads. |
| Calibrated model | Metric observations and justified parameter fitting, with validation on relevant conditions beyond a fitted trajectory. |

These are reporting claims, not a mandatory checklist or a single CLI status enum. Validate the level needed by the current request. `execution_passed`, a finite state, a draggable cloth or a high silhouette IoU establishes only its own evidence. Keep input integrity, appearance, execution, mechanics and calibration distinct. State the strongest supported result and link its asset/UI, open command and reports; name any unfinished behavior that affects the user's intended use.

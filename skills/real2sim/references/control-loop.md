# Scope-driven control loop

This is the external coding agent's decision policy. The harness stores scope, memory and evidence and supplies execution tools; `real2sim agent` remains a UI launcher. Choose the next action that advances the current user outcome, rather than completing every possible reconstruction stage.

Choose the module that owns a defect: harness policy in `src/robocasa_real2sim`, native scene/physics in `src/robosimstudio`, browser controls in `src/robosimstudio_studio`, live loading/RPC in `src/robosimstudio_real2sim_agent`. These are internal modules of the real2sim installation. Preserve source/license records.

## Restore the current task

Understand the latest message, then restore the existing session once at task entry, continuation after context loss, or a meaningful scope change:

```sh
real2sim session-brief --session <session>
```

The brief is a bounded view of the current goal, relevant memory/evidence, last failure and live state. Use its links or `memory-recall` to resolve a specific gap. Prefer `--compact` for `scene-build`, `scene-run`, `scene-snapshot`, `scene-render`, `ref-frames` and `scene-compare`; full evidence remains on disk. Avoid repeatedly dumping the whole history or complete run reports. Use brief `--probe` when fresh live state is needed and saved liveness is uncertain; a normal continuation does not require probing every surface again. For a task without a session, use the available request/artifacts and create a session only when the workflow needs one.

The latest explicit user instruction wins over an older stored goal or inferred plan. A status question alone does not cancel the task. When the deliverable or constraints change, persist the new focus before taking another dependent action:

```sh
real2sim session-focus --session <session> --mode asset \
  --goal 'Inspect and drag the target asset' --constraint 'No folding runs' \
  --validation interaction
```

Modes are `asset`, `scene`, `interaction` and `tooling`. Validation levels are `unspecified`, `none`, `runtime`, `visual` and `interaction`; record the user's requested level rather than assuming the most expensive one. `none` suppresses additional validation runs, not authoring/preview work necessary to deliver the requested UI. Later instructions can change the level.

`session-focus` records scope and invalidates the old next action. It does not stop services or subprocesses. Stop scheduling unrelated work, notify delegated workers and cancel the agent's own known, unnecessary tasks through their control handles where appropriate. Preserve progress with a lightweight checkpoint and existing artifact paths; do not wait to finish an old candidate or polish its archive before switching. Do not terminate unknown services or the user's work.

## Decision states

New user steering returns to **Scope** from any state. Reconcile old memory with the new focus before resuming; completed evidence is reused only if it still supports the current claim.

| State | Select the next action | Exit condition |
| --- | --- | --- |
| Scope | Identify the current deliverable, constraints and required evidence level from the latest request. | The requested outcome is concrete enough to act on. |
| Restore | Read the brief, reconcile prior work, persist changed focus and invalidate unrelated pending actions. | Current goal and relevant evidence are known. |
| Check / observe | Fill the specific missing fact: inspect an input/frame, dependency status, stale rebuild error or live state. | Enough evidence exists to choose an implementation step. |
| Author | Make the smallest relevant asset, native scene, appearance or tooling change. | A concrete result can be previewed or checked. |
| Preview | Open/reuse the native UI for asset work; expose the target and requested handles promptly. Use saved-state rendering for appearance revisions. | The requested asset or change is reviewable. |
| Validate | Run only checks needed for the current claim and change. Compare relevant states and diagnose a concrete failure. | The claim is supported, or a specific discrepancy justifies revision. |
| Deliver / revise | Deliver the requested result and evidence, or choose a bounded next step supported by new information. | Requested work is complete, or the next useful action is explicit. |

These are decisions, not mandatory command invocations. Reuse a suitable asset UI that is already running. For a tooling change, relevant code checks can replace UI preview; a full simulation is not a prerequisite.

Record meaningful transitions with `session-checkpoint`, using the current focus revision returned by the brief/focus operation:

```sh
real2sim session-checkpoint --session <session> --stage preview \
  --focus-revision <revision> --note 'Native asset UI is available' \
  --evidence <saved-report>
```

Stages are `inspect`, `author`, `preview`, `validate`, `deliver` and `blocked`. Use `--next-action` for a concrete next step within that focus. A delayed worker must refresh after a revision mismatch instead of restoring its obsolete next action. Checkpoints preserve progress, not a requirement to log every tool call.

## Scope and evidence gates

- **Full scene → asset:** stop environment completion, further material research, robot replay and folding candidates that no longer serve the request. Reuse the asset/texture in a minimal scene or suitable live session. Preserve the full-scene work; make the inspectable, draggable asset the first deliverable. Resume extra appearance or mechanics work when it is part of the current request.
- **Appearance only:** reuse a compatible saved trajectory and `model_geometry.npz` with `scene-render`. Change the look and compare it without rerunning physics. Geometry/topology, material or motion changes need their own relevant evidence; an old trajectory cannot validate changed dynamics.
- **Environment checks:** use recorded readiness when dependencies, source resolution, device/renderer choice and relevant context have not changed. Run `runtime-status` when compatibility is unknown or changed, `doctor` when a provider is unknown, and a fresh live probe when its state matters. A scene or internal runtime edit calls for the relevant rebuild/reload and checks; it does not by itself invalidate pinned Newton/RTX dependencies or require reinstalling them. Metadata readiness does not establish GPU or dynamics success.
- **Validation level:** asset inspection/dragging needs the corresponding native UI behavior, not a complete manipulation episode or material calibration. A requested interaction needs relevant motion/contact evidence. A tooling change needs meaningful checks appropriate to that change. Follow current validation constraints; do not convert one task's “no tests” instruction into a permanent rule.

## Memory and failed approaches

Use memory to preserve decisions that change the next action, not to narrate every tool call. Query a small relevant subset when the brief is insufficient:

```sh
real2sim memory-recall --session <session> --query 'cloth dragging' --limit 5
real2sim memory-put --session <session> --key <stable-key> \
  --kind decision --scope focus \
  --content '<choice, reason and applicable scope>' --source inferred
```

Kinds are `goal`, `constraint`, `decision`, `fact` and `failed_approach`; sources are `user`, `observed` and `inferred`. Use `session-focus` for the active goal/scope. `--scope` is `focus` or `session`: goals/constraints default to focus, other kinds to session. Put temporary decisions and retry budgets in focus scope; retain session-wide facts or failed approaches only with their applicable conditions. Record observations with their available `--evidence <path>`; keep hypotheses labeled inferred. A failed approach should record its method, conditions, observed failure and evidence so a resumed agent can avoid repeating it. Successful CLI execution can still be a failed approach to the requested visual/contact result. Reconcile stale or conflicting memory with the current request instead of treating every entry as a permanent instruction.

Before retrying the same failure, identify what changed or what diagnostic will distinguish a cause. Choose and record a per-approach budget appropriate to the current task: runs, time or another useful bound. The budget is configurable through decision memory and user constraints; there is no universal three-attempt rule. An unchanged expensive run without a new diagnostic is not a useful retry.

The budget limits that approach, not the whole task. At its boundary, reassess from retained evidence and choose a different evidence-backed action within scope. Report the best supported result, unresolved work and next useful step without marking an unmet goal complete. Do not silently reset the budget or start another open-ended candidate family. A scope reduction can make an old failure irrelevant to the new deliverable; preserve it without letting it block a working asset UI.

# Editable scene reconstruction

Use this path for editable asset reconstruction and observed-scene iteration. Real2sim owns inputs, operations, native UI and evidence. The coding agent performs observation and revision; integrated scene/UI modules under `src/robosimstudio*` retain compatibility imports from RoboSimStudio. Install and operate them through real2sim. Newton VBD handles flexible cloth, rods and soft volumes. Read [agent.md](agent.md) for architecture and success levels. This workflow does not infer an arbitrary video's rest geometry, material or contact automatically.

## Preserve and inspect the reference

Create a session with `session-init --output <session>`. Register the video, metadata and asset directories using `--reference`; `ref-frames --session <session> <video> --times ...` also registers its source. Inputs are copied with hashes. Choose times before contact, during each manipulation, after release and at the final state. Open the contact sheet and the original frames. Requested FFmpeg seek times are labeled explicitly; they are not measured video PTS.

Keep original dataset identifiers, URLs, revision, license, units and transformations. Preserve original recordings separately from rotated, trimmed or resampled derivatives. A composited or synthetic dataset is not a real joint-contact experiment. Copied requests are verbatim; their dependencies must be copied/registered and paths resolved explicitly.

## Author a native scene

Edit `<session>/scene.py`; use the compatibility import `import robosimstudio as rss` and bind `scene = rss.Scene(...)`. For asset mode, add the target, necessary support/collision and a useful camera, with native manipulation controls. Add the observed table/background, robot and motion only when the task requires that scene or interaction. Asset inspection does not require a full robot episode. Use scene-relative helpers/assets, explicit units and component frames. `BUILD` is an optional dictionary of `Scene.build` options; leave `sim` unbound and let the harness choose device and renderer. Viewer calls are suppressed during loading.

Keep case geometry/routines and assets in the session. Fix harness execution/evidence policy in `src/robocasa_real2sim`; native implementation belongs in `src/robosimstudio`, `src/robosimstudio_studio` or `src/robosimstudio_real2sim_agent` according to responsibility. Preserve source/license records under `third_party/robosimstudio` and asset-specific licenses. Record the workspace revision/diff used by the attempt.

Robot models are optional session inputs. When needed, include the chosen URDF/MJCF, geometry, textures and relative dependencies with source/license metadata. Do not assume that the full upstream robot model catalog is installed or bundle it into an asset-only task.

Inspect the runtime's default cell, ground and room/scenery as part of the scene. Configure them explicitly where necessary so support surfaces do not overlap and display geometry does not obscure the asset. Reuse native cloth panels, handles and gizmos for dragging rather than introducing a separate imitation of the simulator UI.

Cloth needs a triangle mid-surface, soft solids need tetrahedra, and rods need a centerline/cross-section. Record the rest state and parameter provenance. Use measured robot states if provided; retain joint names, gripper scaling and timestamp conventions. Inferred camera/shape parameters remain assumptions until constrained by metric observations. Explicit pin constraints can represent a grasp when the runtime cannot produce a physical pinch, but record their selection rule, attachment/release times, missed grasps and the fact that they replace contact mechanics.

When an authored interaction is needed, bind `routines = {"name": function}` or `routine`. A routine accepts `sim`, steps it once, then yields once per control step. It must not step while being created or while returning. Helpers which set joint/pin targets must not advance or reset the simulator. Release temporary constraints and remove hooks in `finally`. Label prescribed motion/forces and preserve missed grasps; do not teleport all cloth vertices through an animation and call that physical reconstruction. A simple draggable asset needs no folding routine.

## Run, observe and revise

Restore the current goal and relevant evidence with `session-brief --session <session>`; use the [control loop](control-loop.md) to select necessary checks. Check runtime compatibility when it is unknown or changed, then use `agent --session <session> --device cuda:0 --renderer rtx`. The agent entrypoint reuses an existing live UI; otherwise it serves one in the foreground. It neither creates nor resets the scene. Panels default to `drive`; `--panels all` selects the broader workbench. Use `--resume` or the native resume control when stepping is wanted. A paused viewport can be inspected, but dynamic dragging needs the appropriate native controls and stepping. A reused server retains its current state and configuration; inspect the returned status rather than rechecking unchanged dependencies.

`session-serve` remains the direct server entrypoint. `session-ctl --session <session>` forwards native commands to that page: `status`, `snapshot --camera <name>`, `ref frames`, `compare`, `check`, `play <routine> --record <take>`, `reset`, and `log`; its native `--help` describes arguments. Its run directory and socket are selected from the session. Edit and inspect the rebuilt page. A failed rebuild can leave the previous scene visible, so check `last_error` and the scene hash before judging it. Reuse this page instead of launching competing servers.

`scene-build`, `scene-run` and `scene-snapshot` start a fresh isolated worker instead of using the browser's current state. For a requested interaction, use `scene-run --session <session> --routine <name> --steps N --capture-every K --record --camera <name>` with an explicit device/renderer. A normal early routine completion can succeed without filling the step budget; inspect `budget_completed`, `completed_steps`, `routine_status` and actual times. MP4 is constant frame rate; timestamped captures describe a short final interval exactly. Isolation and fingerprints do not guarantee bitwise determinism; record runtime/solver settings and check repeatability when it affects a fitting or validation claim.

For appearance revisions, use saved states instead of repeating physics:

```sh
real2sim scene-render --session <session> --renderer rtx \
  --trajectory <run>/trajectory.npz --model-geometry <run>/model_geometry.npz
```

The trajectory must match the model's rest geometry, topology, particle IDs and rigid layout. Record rigid/joint state for scenes containing moving robots; do not silently invent missing poses. Offline rendering preserves the physical trajectory and does not upgrade the original run's mechanics verdict. Keep its render artifacts separate from the physical run.

Use RTX for visual judgement. Tiled captures show collider surfaces, omit textures/material appearance and do not reliably identify flexible silhouettes. Explicit aligned masks passed to `scene-compare` produce IoU, centroid and area comparisons without resizing/alignment. Record segmentation method and visibility: an amodal projected mesh and a visible color mask are different targets. Threshold-derived masks must be labeled as estimates. Image pairs without masks require visual review, not an invented match score. A high initial-frame IoU does not establish the folded state, correct robot motion or material calibration.

After each relevant revision, inspect image pairs and, for motion tasks, the temporal sequence. Compare the initial pose, contact/motion order, release behavior and final state. Check relevant deformation/contact diagnostics using actual mesh topology; investigate excessive stretch, intersections, drift or collapse even when every state remains finite. For asset-only work, demonstrate the requested inspection/dragging behavior rather than launching an unrelated motion sequence. Stable dynamics alone cannot pass reconstruction.

## Retain the result

Keep editable source and portable mesh/texture/collision/material dependencies, registered originals, parameter assumptions and all attempts. Retain the exact source version used for a selected result, through a commit or source snapshot; a fingerprint identifies content but does not restore changed files. Use supported export/bundle commands when another target is required; a native scene does not automatically become an equivalent USD/MJCF flexible simulation.

`session-status` identifies the latest operation and stale evidence after source edits. `session-log` records durable harness events; native `session-ctl log` records live readings. Old success is superseded by a failed rerun. The storage contract is single-writer per session; coordinate parallel agents through separate sessions or explicit file ownership.

Report separately: source/input integrity, geometry/camera agreement, visual agreement, routine execution, contact/dynamics diagnostics and any real calibration. Mark unfinished items plainly. Neither a screenshot nor `execution_passed: true` means a finished physical twin.

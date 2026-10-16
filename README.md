# Real2Sim agent harness

中文使用教程见 [使用指南](使用指南.md)。

A small, model-independent execution harness for an AI that can read files and run commands. The caller AI inspects language, photos, videos, geometry or robot episodes, authors an asset request, and repairs it using simulator evidence. The harness builds MJCF/USD, executes configured checks, and retains revisions and reports.

## 项目示例 / Showcase

### 双 YAM 搬瓶：真实记录 → 仿真回放

[![双臂搬瓶的重建场景](docs/media/hero.png)](docs/media/sim-top.mp4)

826 帧双臂关节与夹爪记录，在重建场景中通过物理接触回放；该次单 episode 四个瓶子中心入桶。查看 [真实顶视视频](docs/media/real-top.mp4) · [仿真顶视视频](docs/media/sim-top.mp4) · [验证说明](docs/examples.md#双-yam-搬瓶)。

| 木饰面工作区 | 可交互办公椅 |
|---|---|
| [![照片重建的工作区](docs/media/lab-overview.png)](docs/media/lab-interactions.mp4) | [![办公椅 Isaac 演示](docs/media/chair-sim.jpg)](docs/media/chair-sim.mp4) |
| 14 张照片重建场景，包含四台 YAM、升降桌、椅子、抽屉柜与打印设备。[交互视频](docs/media/lab-interactions.mp4) · [场景与验证](docs/examples.md#木饰面工作区) | 静态模型增加升降、转动、后仰和被动脚轮；当前物理检查 **14/16**。[演示视频](docs/media/chair-sim.mp4) · [已通过项目与待修复项](docs/examples.md#可交互办公椅) |
| **三抽屉活动柜** | **照片驱动的门机构** |
| [![三抽屉活动柜](docs/media/drawer-open.png)](docs/examples.md#三抽屉活动柜) | [![门机构 Isaac 演示](docs/media/door-sim.jpg)](docs/media/door-sim.mp4) |
| 两张照片建立三个被动滑轨、四组转向脚轮与八个独立轮胎；MuJoCo / Isaac 各 **15/15** 检查通过。[参考照片](docs/media/drawer-reference.jpg) · [验证说明](docs/examples.md#三抽屉活动柜) | 根据开、关照片建立门扇与旋转铰链，验证开合与角度限位。[开合视频](docs/media/door-sim.mp4) · [验证说明](docs/examples.md#照片驱动的门机构) |

以上为已保存的项目结果；尺寸、质量、摩擦和相机参数未完成实物标定。单次回放和配置检查的通过范围见 [示例说明与报告](docs/examples.md)。

<details>
<summary>柔性探索：Newton 袋体接触与服装网格</summary>

| 袋体与刚体接触 | T 恤几何预览 |
|---|---|
| [![Newton 袋体接触示例](docs/media/bag.png)](docs/media/bag.mp4) | ![Newton 官方 T 恤网格](docs/media/shirt.png) |
| Newton 官方 VBD 示例复现，300 帧仿真及官方检查通过。[视频](docs/media/bag.mp4) | 官方网格的单位与坐标转换展示，未执行衣物物理仿真。 |

这些探索尚未接入统一 harness 的柔性重建闭环。参见 [来源与状态](docs/examples.md#柔性探索)。

</details>

## Install

Use Python 3.11 or newer. From the source folder, `./bootstrap.sh` creates `.venv` and installs the authoring/video providers. Use `.venv/bin/real2sim` directly, or activate `.venv` before calling `real2sim`. For an existing virtual environment:

```bash
python -m pip install '.[mujoco,usd,video,episode]'
real2sim capabilities
real2sim doctor --require mujoco --require usd
```

The wheel includes the skill and Isaac support scripts under the Python environment's `share/real2sim`; a checkout is unnecessary after normal wheel installation. `capabilities` reports the actual paths; `REAL2SIM_REPO` can explicitly select a checkout instead. Run Isaac scripts with the simulator's own Python, selected through `REAL2SIM_ISAAC_PYTHON`, `--isaac-python` or the workflow's `runtime.python`. The current Isaac backend targets 5.x. Complete NVIDIA's first-launch licence confirmation yourself. Install COLMAP separately when photo reconstruction is needed.

## Give it to an AI

Ask the AI to run `real2sim capabilities`, read the reported `skill`, and follow it with your inputs, requested interactions and simulator. Any agent with filesystem and command access can use the same contract. It needs no harness-specific model or agent server.

Create a task from supplied files or just a language goal. `--input` is repeatable; `--episode` attaches a robot episode:

```bash
real2sim task-init --input INPUT_PATH --goal "ASSET AND REQUIRED INTERACTIONS" \
  --simulator both --output TASK
```

The task handoff is `task.json` plus `AGENT_TASK.md`. The AI writes `request.json` and `workflow.json`, then runs:

```bash
real2sim pipeline --plan TASK/workflow.json --output TASK/run
real2sim task-status --task TASK
```

Task initialization retrieves similar projects into `references.json`; managed runs automatically retain successful and failed attempts, parameter revisions and test evidence in a local SQLite experience store. The agent inspects relevant conditions with `experience-show`, adapts useful parts and validates the new asset. Search works with bilingual keywords and mechanism/test features, without a model service:

```bash
real2sim experience-search --query "drawer caster rolling" --simulator isaac
real2sim experience-show REFERENCE_ID
real2sim experience-import EXISTING_RUN_OR_ASSET
```

Set `REAL2SIM_MEMORY` to select/share the database; its default is `~/.local/share/real2sim/experience.sqlite3` (or under `XDG_DATA_HOME`). Historical passes are scoped to their recorded attempt; legacy assets are reference-only. This improves the agent's context through accumulated project evidence. See the [contracts](skills/real2sim/references/harness.md#project-experience) for annotations and workflow settings.

See the [skill](skills/real2sim/SKILL.md) for the agent loop and [contracts](skills/real2sim/references/harness.md) for exact fields.

## Scope

- Inputs: language goals, photos, videos, white models/meshes, existing USD/MJCF, and timestamped video-plus-trajectory robot episodes.
- Authoring: explicit rigid-body trees with fixed, revolute, continuous and prismatic joints; a single-file program adapter handles custom authoring.
- Outputs: portable MJCF/USD and manifests; current MuJoCo/Isaac validation reports for configured tests.
- Geometry: fixed metric source evidence and localized candidate comparisons.
- Execution: resumable stages, bounded parameter revisions, timings and retained failure evidence.

The caller AI supplies semantic interpretation, dimensions and mechanics. Hidden structures, metric scale and physical parameters require evidence or declared assumptions. MuJoCo episodes use a configured PD controller; Isaac episodes replay joint targets using an authored USD scene's drives and contacts. Automatic object pose tracking and physical parameter identification remain outside the harness. `both` exports two formats; each simulator needs its own test run. Successful compilation or a geometry score does not validate dynamics.

Isaac episode runs use the existing `simulate` command and save `trajectory.npz` and a render-only `episode_replay.usdc`. Export cameras separately, without rerunning physics or tests:

```bash
real2sim export-videos RUN/episode_replay.usdc --output RUN/videos \
  --camera /World/TopCamera 640 480 --camera /World/PanoramaCamera 1920 1080 \
  --isaac-python /path/to/isaac/python.sh
```

Repeat `--camera PRIM WIDTH HEIGHT` for additional cameras; `--frames N` exports a short preview. See the [episode contract](skills/real2sim/references/harness.md#robot-episodes) for multi-articulation joint mappings and the supported test scope.

The design draws on [USDCraft](https://xingyoujun.github.io/usdcraft/) for programmable assets and geometry feedback, and [Agentic Real2Sim](https://github.com/agentic-real2sim/agentic_real2sim) for simulator feedback with robot episodes. It is an independent implementation with the scope above.

## Develop

```bash
python -m pip install -e '.[dev,usd,mujoco,video,episode]'
ruff check src scripts tests
ruff format --check --quiet src scripts tests
python -m pytest
python -m build
```

CPU tests do not establish Isaac runtime compatibility. Read [AGENTS.md](AGENTS.md) before changing the harness.

Source lives in `src/robocasa_real2sim`, Isaac adapters in `scripts`, and shared contracts in `skills/real2sim`. To format changes, run `ruff format src scripts tests`. GitHub Actions checks formatting, CPU tests and wheel installation on Python 3.11 and 3.12.

## Upload to GitHub

Use this source directory as the repository root. Local environments, input data, task runs, caches and build output are ignored; keep large assets and generated videos outside the source repository. After creating an empty repository on GitHub, run the following from this directory, replacing `YOUR_ACCOUNT/YOUR_REPOSITORY` with its actual location:

```bash
git init -b main  # Only needed if this directory is not already a Git repository.
git add .
git diff --cached --stat
git commit -m "Initial import"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

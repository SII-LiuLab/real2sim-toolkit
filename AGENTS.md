# Real2Sim Toolkit

For real2sim workflow tasks, read `skills/real2sim/SKILL.md`. The scope includes household objects, furniture and factory equipment. Preserve the user's requested simulator and physical interaction.

Implementation lives in `src/robocasa_real2sim`; `real2sim` is the execution CLI. Isaac entrypoints live in `scripts` and use Isaac's Python. Keep simulator imports inside runtime functions, after `SimulationApp` initialization, so ordinary CPU tooling and CLI help work without Isaac.

Use explicit units, physical states, component frames and parameter provenance. Preserve relative asset dependencies. Distinguish reconstructed appearance, authored mechanics, structural checks and actual dynamics validation. Failed reruns must replace old success reports.

Tests: `python -m pytest`. Native fixture integration needs separately installed RoboCasa; USD tests need the `usd` extra. CPU CI does not establish Isaac compatibility. Current adapters target 5.x PhysX; test API/backend migrations on the target runtime.

The repository contains tools and instructions. Do not add demonstration scenes or example assets unless the user requests them.

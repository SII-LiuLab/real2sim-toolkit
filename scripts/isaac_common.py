"""Shared runtime setup. Import Isaac/Omniverse only after SimulationApp."""

import json
from pathlib import Path
import sys


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def start_app(headless):
    try:
        from isaacsim import SimulationApp
    except ImportError as error:
        raise RuntimeError("Run with your Isaac Sim installation's python.sh") from error
    return SimulationApp({"headless": headless})


def runtime_version():
    from isaacsim.core.version import get_version
    version = get_version()
    if version[2] != "5":
        raise RuntimeError(f"This runtime adapter targets Isaac Sim 5.x / PhysX; detected {version[0]}. Adapt and test the backend before using another major version.")
    return version[0]


def add_toolkit_path():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def wait_for_stage(app, context):
    for _ in range(600):
        app.update()
        if context.get_stage() and context.get_stage_loading_status()[2] == 0:
            return context.get_stage()
    raise RuntimeError("Stage failed to finish loading within 600 updates")

"""Run RoboSimStudio's existing live workbench over a harness session."""

from pathlib import Path
from hashlib import sha256
from contextlib import contextmanager
import os
import tempfile
import time

from .io import write_json
from .rss_compat import control_api, evict_live_helpers, live_server_api
from .session import begin_operation, fail_operation, finish_operation, session_status


def control_session(session, arguments):
    """Use RoboSimStudio's native controller against this session's live page."""
    from .io import read_json

    build_parser, main = control_api()
    root = Path(session).resolve()
    latest = session_status(root)["operations"].get("scene-serve")
    if not latest:
        raise ValueError("Start this session with session-serve first")
    path = root / latest["directory"] / "live.json"
    if not path.is_file():
        raise ValueError("This session's live page has not started; inspect scene-serve status")
    live = read_json(path)
    if live.get("status") != "running":
        raise ValueError("This session's live page is stopped; start session-serve again")
    if any(arg in {"--sock", "--run-dir"} or arg.startswith(("--sock=", "--run-dir=")) for arg in arguments):
        raise ValueError("session-ctl selects the session socket and run directory automatically")
    argv = ["--run-dir", str(path.parent), "--sock", live["socket"], *arguments]
    parsed = build_parser().parse_args(argv)
    if parsed.cmd == "up":
        raise ValueError("Use session-serve to start a page with the harness's Newton backend policy")
    return main(argv)


@contextmanager
def _launch_guard(root):
    """Serialize page startup/lifetime without placing lock files in assets."""
    import fcntl
    path = Path(tempfile.gettempdir()) / ("real2sim-" + sha256(str(root).encode()).hexdigest()[:16] + ".launch.lock")
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This session already has a page starting or running; use real2sim agent to reconnect") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def serve_session(session, *, port=8080, device="cuda:0", renderer="tiled", panels="all", resume=False):
    """Start one page per session; an existing page must be reused explicitly."""
    from .agent import live_agent_status
    root = Path(session).resolve()
    with _launch_guard(root):
        current = live_agent_status(root)
        if current is not None:
            raise RuntimeError(f"This session is already served at {current['url']}; use real2sim agent --session {root} to reconnect")
        return _serve_session(root, port=port, device=device, renderer=renderer, panels=panels, resume=resume)


def _serve_session(session, *, port=8080, device="cuda:0", renderer="tiled", panels="all", resume=False):
    """Serve the editable session, rebuilding on save with flexible VBD routing."""
    from .rss_backend import load_scene
    from .runtime_env import assert_studio_runtime

    runtime_environment = assert_studio_runtime(renderer=renderer)
    native = live_server_api()
    root = Path(session).resolve()
    session_status(root)
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if panels not in {"all", "drive"}:
        raise ValueError("panels must be all or drive")
    started = begin_operation(root, "scene-serve", {"port": port, "device": device, "renderer": renderer,
                                                   "panels": panels, "resume": resume, "pid": os.getpid()})
    output = root / started["directory"]
    # UNIX domain socket paths have a small platform limit (108 bytes on Linux).
    socket = Path(tempfile.gettempdir()) / ("real2sim-" + sha256(str(root).encode()).hexdigest()[:16] + ".sock")

    class SessionServer(native.LiveServer):
        def load(self):
            attempt = begin_operation(root, "live-build", {"device": device, "renderer": renderer})
            try:
                evict_live_helpers(self)
                runtime = load_scene(self.path, device=device, renderer=renderer,
                                     build_overrides={"control_mode": "joint_position"})
                self.harness_runtime = runtime
                report = runtime.build_report()
                if not report["finite"]["passed"]:
                    raise RuntimeError("The built scene has a non-finite initial state")
                report_path = root / attempt["directory"] / "build.json"
                write_json(report_path, report)
                finish_operation(root, attempt["id"], report, {"report": report_path})
                return native.Loaded(ns=runtime.namespace, scene=runtime.scene, sim=runtime.sim,
                                     seconds=runtime.build_seconds)
            except Exception as error:
                fail_operation(root, attempt["id"], str(error))
                raise

        def open_page(self):
            super().open_page()
            report = {"status": "running", "url": self.url, "scene": "scene.py",
                      "run_dir": output.relative_to(root).as_posix(), "device": device,
                      "renderer": renderer, "panels": panels, "build_error": self.last_error,
                      "socket": str(socket), "pid": os.getpid(),
                      "runtime_environment": runtime_environment,
                      "reconstruction_validated": False}
            write_json(output / "live.json", report)
            if self.last_error:
                fail_operation(root, started["id"], self.last_error, report, {"live": output / "live.json"})
            else:
                finish_operation(root, started["id"], report, {"live": output / "live.json"})
            print(f"Real2Sim workbench: {self.url}", flush=True)

    config = native.StudioConfig(port=port, share=False, panels=panels, live=False,
                                out=str(output / "studio"))
    server = SessionServer(scene_path=root / "scene.py", run_dir=output,
                           sock_path=socket, config=config, watch=True, hold=not resume)
    try:
        server.serve()
    except Exception as error:
        state = session_status(root)["operations"].get("scene-serve", {})
        if state.get("id") == started["id"] and state.get("status") == "running":
            fail_operation(root, started["id"], str(error))
        elif state.get("id") == started["id"]:
            failed = begin_operation(root, "scene-serve", {"port": port, "device": device, "renderer": renderer})
            fail_operation(root, failed["id"], str(error))
        raise
    finally:
        path = output / "live.json"
        if path.exists():
            from .io import read_json
            report = read_json(path)
            report.update(status="stopped", stopped_at_unix=time.time())
            write_json(path, report)

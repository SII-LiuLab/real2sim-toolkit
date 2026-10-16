"""One entry point for a coding agent's editable reconstruction workbench.

The language model runs in the calling coding agent. This module describes its
tools and opens or reconnects to the native scene UI without resetting it.
"""

from pathlib import Path

from .io import read_json
from .session import session_status


def agent_contract():
    """Return the tool contract without importing a simulator or starting one."""
    return {
        "name": "real2sim",
        "role": "coding-agent harness for reconstructing interactive simulation assets",
        "entrypoint": "real2sim agent --session SESSION",
        "model_execution": "external coding agent using skills/real2sim/SKILL.md",
        "ownership": {
            "real2sim": "references, reconstruction decisions, assets, sessions and evidence",
            "robosimstudio": "internal adapted scene, flexible-domain, UI and recording modules in this real2sim distribution",
            "flexible_physics": "Newton VBD",
        },
        "tools": {
            "restore_context": "real2sim session-brief --session SESSION",
            "change_scope": "real2sim session-focus --session SESSION --mode asset|scene|interaction|tooling --goal GOAL",
            "checkpoint": "real2sim session-checkpoint --session SESSION --stage STAGE --focus-revision N --note NOTE",
            "remember": "real2sim memory-put --session SESSION --key KEY --kind decision --source observed --content TEXT",
            "recall": "real2sim memory-recall --session SESSION --query QUERY --limit 5",
            "environment": "real2sim runtime-status --renderer tiled|rtx",
            "initialize": "real2sim session-init --output SESSION --reference INPUT",
            "inspect_reference": "real2sim ref-frames --session SESSION INPUT --times 0 3 6",
            "edit": "Author SESSION/scene.py and local assets using the native rss.Scene API",
            "interact": "real2sim agent --session SESSION --panels drive",
            "control": "real2sim session-ctl --session SESSION status|resume|pause|reset|snapshot|play|log",
            "execute": "real2sim scene-run --session SESSION --routine NAME --steps N",
            "render_recording": "real2sim scene-render --session SESSION --trajectory TRAJECTORY --model-geometry GEOMETRY --record",
            "compare": "real2sim scene-compare --session SESSION --reference IMAGE --rendered IMAGE",
            "evidence": "real2sim session-status --session SESSION",
            "portable_flexible_asset": "real2sim build-deformable --request REQUEST --output ASSET",
        },
        "loop": ["restore current scope and usable evidence", "inspect only missing facts", "author geometry, materials and interaction",
                 "inspect and drag in UI", "capture, compare and revise", "deliver asset and evidence"],
        "context_policy": {
            "default": "Use session-brief once at entry/resume or a scope change; load full reports only for a specific gap.",
            "steering": "Record the current focus before dependent work; an old checkpoint cannot supply its next action after scope changes.",
            "response_size": "Use --compact on scene operations; full reports and recordings remain on disk.",
            "checks": "Inspect environment/live state when needed or changed; recorded metadata readiness is not a fresh device test.",
        },
        "completion": {
            "prepared": "geometry and declared parameters exist",
            "execution": "requested runtime operation completed",
            "appearance": "reference comparison reviewed separately",
            "physics": "required interaction checked separately from appearance",
            "calibration": "requires measured data and independent validation",
        },
        "limitations": [
            "No embedded LLM or automatic reconstruction of arbitrary video.",
            "Authored scale, materials and constraints must retain their provenance.",
            "A successfully executed scene is not a calibrated physical twin.",
        ],
    }


def live_agent_status(session):
    """Probe the session's recorded native socket; never mistake busy for absent."""
    from .rss_compat import control_request, LiveServerUnavailable

    root = Path(session).resolve()
    manifest = session_status(root)
    attempt = manifest["operations"].get("scene-serve")
    if not attempt:
        return None
    live_path = root / attempt["directory"] / "live.json"
    if not live_path.is_file():
        if attempt["status"] == "running":
            pid = attempt.get("parameters", {}).get("pid")
            if isinstance(pid, int) and pid > 0:
                import os
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return None
            raise RuntimeError("This session is already starting; inspect session-status before opening another page")
        return None
    live = read_json(live_path)
    if live.get("status") == "stopped":
        return None
    socket_path = live.get("socket")
    if not socket_path:
        raise RuntimeError("The live page record has no control socket; inspect its scene-serve attempt")
    try:
        status = control_request(socket_path, {"cmd": "status"}, timeout=3.)
    except LiveServerUnavailable as error:
        # The native transport also wraps access errors as unavailable. Only a
        # missing socket/refused connection means it is safe to start a page.
        cause = error
        while cause.__cause__ is not None:
            cause = cause.__cause__
        import errno
        if isinstance(cause, OSError) and cause.errno in {errno.ENOENT, errno.ECONNREFUSED}:
            return None
        raise
    if not status.get("ok"):
        raise RuntimeError(f"The live page rejected status: {status.get('error', status)}")
    if Path(status.get("file", "")).resolve() != root / "scene.py":
        raise RuntimeError("The session socket belongs to another scene; refusing to reuse or replace it")
    return {"status": "failed" if status.get("last_error") or not status.get("finite", False) else "ready",
            "session": str(root), "url": status["url"], "reused": True,
            "socket": socket_path, "live": status, "scene_source_stale": bool(attempt.get("stale")),
            "reconstruction_validated": False}


def open_agent(session, *, port=8080, device="cuda:0", renderer="tiled", panels="drive", resume=False):
    """Keep a live session intact, or start its page in the foreground."""
    from .runtime_env import assert_studio_runtime
    from .rss_compat import control_request

    runtime = assert_studio_runtime(renderer=renderer)
    current = live_agent_status(session)
    if current is not None:
        if resume and current["status"] == "ready":
            reply = control_request(current["socket"], {"cmd": "resume"})
            if not reply.get("ok"):
                raise RuntimeError(f"Cannot resume the live page: {reply.get('error', reply)}")
            current["live"]["paused"] = bool(reply["paused"])
        current["runtime_environment"] = runtime
        current["note"] = "Reused the running page; its current scene, view settings and physics state were preserved."
        return current
    from .live import serve_session
    serve_session(session, port=port, device=device, renderer=renderer, panels=panels, resume=resume)
    return {"status": "stopped", "session": str(Path(session).resolve())}

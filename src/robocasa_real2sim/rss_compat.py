"""The optional RoboSimStudio compatibility boundary.

Private camera lookup, the experimental live-page packages and the narrowly
versioned OVRTX bridge belong here. This module imports no simulator at module
load and never steps/resets a scene, changes a solver, or selects a renderer.
Unsupported API shapes fail with installed versions rather than falling back
to a different runtime or silently patching the reference checkout.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import inspect
import math
import os
from pathlib import Path
import re
import sys


class RSSCompatibilityError(RuntimeError):
    """The installed optional runtime does not expose the expected API."""


class LiveServerUnavailable(ConnectionError):
    """Native NoServer: connection failed; inspect the preserved OS cause."""


class LiveServerNoReply(TimeoutError):
    """A page did not answer; this does not establish that its server is gone."""


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _incompatible(message):
    versions = ", ".join(f"{name}={_version(name) or 'not installed'}"
                         for name in ("robocasa-real2sim", "newton", "warp-lang", "ovrtx"))
    return RSSCompatibilityError(f"RoboSimStudio compatibility error: {message}. "
                                 f"Installed: {versions}. Use the pinned runtime or update rss_compat.py explicitly.")


def _callable(owner, name, label):
    value = getattr(owner, name, None)
    if not callable(value):
        raise _incompatible(f"{label}.{name} is unavailable")
    return value


def _signature(function, label, *args, **kwargs):
    try:
        inspect.signature(function).bind(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise _incompatible(f"{label} no longer accepts the expected arguments") from error


@contextmanager
def scene_directory(path):
    """Resolve the editable scene's local imports/assets on its worker thread."""
    path = Path(path)
    previous, old_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(path.parent)
        sys.path.insert(0, str(path.parent))
        yield
    finally:
        os.chdir(previous)
        sys.path[:] = old_path


def rss_module():
    """Import the optional Scene/Sim front door only when a runtime is needed."""
    try:
        import robosimstudio as rss
        rss.Scene  # Resolve its lazy runtime export here.
    except ImportError as error:
        raise ImportError("Real2Sim's native runtime dependencies are unavailable; install this repository "
                          "with requirements/studio.txt and requirements/studio-overrides.txt in this Python environment") from error
    except AttributeError as error:
        raise _incompatible("robosimstudio.Scene is unavailable") from error
    return rss


@contextmanager
def suppress_scene_viewers(rss):
    """Use the native scene-loader convention without opening a second page."""
    doors = {name: _callable(rss, name, "robosimstudio") for name in ("view", "draw", "viewer")}
    try:
        for name in doors:
            setattr(rss, name, lambda *args, **kwargs: None)
        yield
    finally:
        for name, value in doors.items():
            setattr(rss, name, value)


def install_renderer_bridge(sim):
    """Keep Newton 1.6 on the known OVRTX 0.4.0 legacy USD/attribute API.

    OVRTX 0.4.0's separate ovstage library can conflict with its bundled stage
    writer. Other SDK versions keep the native path. The guard is evaluated
    when RTX is actually requested so an unrelated RTX install cannot block
    headless physics or tiled rendering. No reference package is modified.
    """
    native = _callable(sim, "make_renderer", "Sim")
    installed = getattr(native, "_real2sim_renderer_bridge", None)
    if installed is not None:
        return installed
    sdk = _version("ovrtx")
    if sdk is None or re.match(r"^0\.4\.0(?:[.+-]|$)", sdk) is None:
        return "native"
    bridge = "ovrtx_0.4.0_legacy_api"

    @wraps(native)
    def make_renderer(*args, **kwargs):
        renderer = native(*args, **kwargs)
        if getattr(renderer, "name", "") != "rtx":
            return renderer
        try:
            newton = _version("newton")
            if newton is None or re.match(r"^1\.6(?:\.|$)", newton) is None:
                raise _incompatible("the OVRTX 0.4.0 legacy bridge is supported only with Newton 1.6.x")
            viewer = getattr(renderer, "viewer", None)
            if viewer is None or not isinstance(getattr(viewer, "_use_ovstage", None), bool):
                raise _incompatible("RTX viewer._use_ovstage no longer exposes the Newton 1.6 bridge")
            viewer._use_ovstage = False
        except RSSCompatibilityError:
            try:
                renderer.close()
            except Exception:
                pass  # Preserve the compatibility failure, not cleanup noise.
            raise
        return renderer

    make_renderer._real2sim_renderer_bridge = bridge
    sim.make_renderer = make_renderer
    return bridge


def resolve_camera(sim, name=None):
    """Resolve a named/default native CameraSpec through the pinned private API."""
    lookup = _callable(sim, "_camera_for", "Sim")
    _signature(lookup, "Sim._camera_for", name)
    camera = lookup(name)  # Preserve native unknown-camera/user-input errors.
    for field in ("name", "width", "height", "intrinsics", "world_matrix", "parent_body"):
        if not hasattr(camera, field):
            raise _incompatible(f"Sim._camera_for returned a camera without {field}")
    _callable(camera, "scaled", "CameraSpec")
    return camera


def offline_renderer_type():
    """Resolve the native replay renderer without importing it in CPU tooling."""
    try:
        from robosimstudio.core.render.offline import OfflineRenderer
    except ImportError as error:
        raise _incompatible("native OfflineRenderer is unavailable") from error
    _signature(OfflineRenderer, "OfflineRenderer", None, None, strict=True)
    for name in ("replay", "close"):
        _callable(OfflineRenderer, name, "OfflineRenderer")
    return OfflineRenderer


@dataclass(frozen=True)
class LiveServerAPI:
    Loaded: type
    LiveServer: type
    StudioConfig: type


def live_server_api():
    """Resolve the experimental RSS live-page extension points in one place."""
    try:
        from robosimstudio_real2sim_agent.scenefile import Loaded
        from robosimstudio_real2sim_agent.server import LiveServer
        from robosimstudio_studio.session import StudioConfig
    except ImportError as error:
        raise _incompatible("native live-page packages are unavailable") from error
    _signature(Loaded, "Loaded", ns={}, scene=None, sim=None, seconds=0.)
    _signature(StudioConfig, "StudioConfig", port=8080, share=False, panels="all", live=False, out="")
    _signature(LiveServer, "LiveServer", scene_path=Path("scene.py"), run_dir=Path("."),
               sock_path=Path("live.sock"), config=None, watch=True, hold=True)
    for name in ("load", "open_page", "serve", "evict_helpers"):
        _callable(LiveServer, name, "LiveServer")
    return LiveServerAPI(Loaded, LiveServer, StudioConfig)


def evict_live_helpers(server):
    """Use the native reload hook; silently retaining old helpers is unsafe."""
    evict = _callable(server, "evict_helpers", "LiveServer")
    _signature(evict, "LiveServer.evict_helpers")
    return evict()


def control_api():
    """Return native ctl parser/main; harness socket/policy remains in live.py."""
    try:
        from robosimstudio_real2sim_agent.ctl import build_parser, main
    except ImportError as error:
        raise _incompatible("native live controller is unavailable") from error
    _signature(build_parser, "ctl.build_parser")
    _signature(main, "ctl.main", [])
    return build_parser, main


def control_request(socket_path, payload, timeout=3.0):
    """Send one native live RPC, preserving absence versus no-reply semantics.

    No retry, startup or socket deletion occurs here. ``LiveServerNoReply``
    includes a busy/full connection backlog and a dropped connection, so it
    must never trigger an automatic second server. ``LiveServerUnavailable``
    preserves native NoServer and its original OS error through ``__cause__``;
    callers can distinguish a missing/refused socket from a permission error.
    Native ``ok=False`` replies are returned unchanged.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a native control request dictionary")
    if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0.:
        raise ValueError("timeout must be positive and finite")
    try:
        from robosimstudio_real2sim_agent.wire import NoReply, NoServer, request
    except ImportError as error:
        raise _incompatible("native live control wire API is unavailable") from error
    _signature(request, "wire.request", socket_path, payload, float(timeout))
    try:
        reply = request(socket_path, payload, float(timeout))
    except NoServer as error:
        raise LiveServerUnavailable(str(error)) from error
    except NoReply as error:
        raise LiveServerNoReply(str(error)) from error
    if not isinstance(reply, dict):
        raise _incompatible("native live control reply is not a dictionary")
    return reply

"""Register unchanged images or extract review frames from a reference video.

This tool does not infer scale, camera calibration, rest shape or material.
FFmpeg is needed only for videos; no simulator is imported.
"""

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from .io import write_json


def _file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_video(path):
    probe = shutil.which("ffprobe")
    if not probe:
        return {"probe_available": False}
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,duration,avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if result.returncode:
        return {"probe_available": True, "probe_error": result.stderr[-2000:]}
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    stream = streams[0] if streams else {}
    metadata = {"probe_available": True, "stream": stream}
    for value in (stream.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            metadata["duration_s"] = duration
            break
    return metadata


def _contact_sheet(frames, target):
    from PIL import Image, ImageDraw, ImageOps

    columns = min(4, len(frames))
    width, height = 256, 210
    sheet = Image.new("RGB", (columns * width, math.ceil(len(frames) / columns) * height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        with Image.open(frame["path"]) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((width, height - 24))
        x, y = index % columns * width, index // columns * height
        sheet.paste(preview, (x, y))
        timestamp = frame["time_s"]
        label = f"{index}: {timestamp:.3f} s" if timestamp is not None else f"{index}: still image"
        draw.text((x + 4, y + height - 20), label, fill="black")
    sheet.save(target)


def extract_reference(source, output_dir, *, times_s=None, fps=1.0, max_frames=12):
    """Copy an image unchanged, or extract video frames into a new evidence folder.

    Explicit ``times_s`` are requested seek times, not recovered frame PTS.
    Otherwise sampling spans the clip (when duration is known), capped by
    ``fps`` and ``max_frames``. Returned frame paths are absolute. Each call
    gets a separate frame folder so previously registered references survive.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "failed", "frames": [],
              "geometry_reconstructed": False, "real_world_calibrated": False}
    report_path = output / "reference.json"
    if Path(source).resolve() == report_path:
        raise ValueError("Reference source must not be the output report path")
    write_json(report_path, report)
    try:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Reference not found: {source}")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        explicit_times = times_s is not None
        if explicit_times:
            times_s = [float(value) for value in times_s]
            if not times_s or len(times_s) > max_frames:
                raise ValueError("times_s must be nonempty and contain at most max_frames entries")
            if any(not math.isfinite(value) or value < 0 for value in times_s):
                raise ValueError("times_s must contain finite nonnegative seconds")
            if any(a >= b for a, b in zip(times_s, times_s[1:])):
                raise ValueError("times_s must be strictly increasing")
        report.update(source=str(source), source_sha256=_file_digest(source))
        try:
            with Image.open(source) as image:
                image.verify()
            is_image = True
        except UnidentifiedImageError:
            is_image = False
        frames_dir = Path(tempfile.mkdtemp(prefix="frames-", dir=output))
        if is_image:
            if explicit_times:
                raise ValueError("times_s is only supported for videos")
            # Keep original bytes, including EXIF; comparison orients a copy.
            target = frames_dir / ("frame_000000" + source.suffix)
            shutil.copyfile(source, target)
            report["source_type"] = "image"
            report["frames"] = [{"path": str(target), "time_s": None}]
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("Video frame extraction requires ffmpeg on PATH")
            metadata = _probe_video(source)
            report.update(source_type="video", video=metadata)
            duration = metadata.get("duration_s")
            if explicit_times and duration is not None and times_s[-1] >= duration:
                raise ValueError("Requested time is outside the video duration")
            if not explicit_times:
                step = max(1 / fps, duration / max_frames) if duration else 1 / fps
                times_s = [index * step for index in range(max_frames)
                           if duration is None or index * step < duration]
            for index, timestamp in enumerate(times_s):
                target = frames_dir / f"frame_{index:06d}.png"
                result = subprocess.run(
                    [ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{timestamp:.9f}",
                     "-i", str(source), "-map", "0:v:0", "-frames:v", "1", str(target)],
                    capture_output=True, text=True, check=False, timeout=120,
                )
                if result.returncode:
                    raise RuntimeError(f"ffmpeg failed at {timestamp:g} s: {result.stderr[-2000:]}")
                if not target.is_file():
                    if explicit_times:
                        raise ValueError(f"No video frame decoded at requested time {timestamp:g} s")
                    break
                report["frames"].append({"path": str(target), "time_s": timestamp,
                                         "timestamp_kind": "requested_seek"})
            if not report["frames"]:
                raise ValueError("No reference frames decoded")
        for frame in report["frames"]:
            with Image.open(frame["path"]) as image:
                image.load()
                frame["width"], frame["height"] = ImageOps.exif_transpose(image).size
            frame["sha256"] = _file_digest(frame["path"])
        sheet_path = frames_dir / "contact_sheet.png"
        _contact_sheet(report["frames"], sheet_path)
        report.update(status="prepared", contact_sheet=str(sheet_path), notes=[
            "Reference images are preserved; no scale, rest shape or material is inferred.",
            "Video timestamps are requested seek times; synchronize comparison frames explicitly.",
        ])
        write_json(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        write_json(report_path, report)
        raise

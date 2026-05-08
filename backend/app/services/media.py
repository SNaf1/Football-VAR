from __future__ import annotations

import math
import subprocess
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np
from fastapi import UploadFile

from ..config import Settings


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def relative_media_url(path: Path, settings: Settings) -> str:
    if path.is_relative_to(settings.media_path):
        return "/media/" + path.relative_to(settings.media_path).as_posix()
    if path.is_relative_to(settings.sample_clips_path):
        return "/samples/" + path.relative_to(settings.sample_clips_path).as_posix()
    raise ValueError(f"Cannot create public URL for path {path}")


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video at {path}")
    return capture


def get_video_metadata(path: Path) -> dict[str, float | int]:
    capture = _open_video(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration = frame_count / fps if fps else 0
    return {"fps": fps, "duration": duration, "width": width, "height": height}


def save_upload_file(upload: UploadFile, destination_dir: Path) -> Path:
    extension = Path(upload.filename or "clip.mp4").suffix.lower() or ".mp4"
    file_path = destination_dir / f"original{extension}"
    with file_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return file_path


def extract_frame(path: Path, timestamp: float) -> tuple[np.ndarray, float]:
    capture = _open_video(path)
    safe_timestamp = max(0.0, timestamp - 0.001)
    capture.set(cv2.CAP_PROP_POS_MSEC, safe_timestamp * 1000)
    ok, frame = capture.read()
    actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not extract frame at {timestamp} seconds from {path}")
    return frame, actual


def write_image(path: Path, frame: np.ndarray) -> None:
    cv2.imwrite(str(path), frame)


def create_poster(video_path: Path, destination: Path) -> Path:
    frame, _ = extract_frame(video_path, 0)
    write_image(destination, frame)
    return destination


def extract_clip(video_path: Path, output_path: Path, start_seconds: float, end_seconds: float) -> dict[str, float]:
    ffmpeg_duration = max(0.04, float(end_seconds) - float(start_seconds))
    if _extract_clip_with_ffmpeg(video_path, output_path, start_seconds, ffmpeg_duration):
        metadata = get_video_metadata(output_path)
        metadata["duration"] = round(min(ffmpeg_duration, float(metadata["duration"]) or ffmpeg_duration), 3)
        return metadata
    return _extract_clip_with_opencv(video_path, output_path, start_seconds, end_seconds)


def _extract_clip_with_ffmpeg(video_path: Path, output_path: Path, start_seconds: float, duration: float) -> bool:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    base_command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
    ]
    commands = [
        base_command
        + [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_path),
        ],
        base_command + ["-c", "copy", str(output_path)],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=True, capture_output=True)
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            continue
    return False


def _extract_clip_with_opencv(video_path: Path, output_path: Path, start_seconds: float, end_seconds: float) -> dict[str, float]:
    capture = _open_video(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(math.floor(start_seconds * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_seconds * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    current = start_frame
    while current < end_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        current += 1

    writer.release()
    capture.release()
    clip_duration = max(0.04, (current - start_frame) / fps)
    return {"fps": fps, "duration": clip_duration, "width": width, "height": height}


def sample_video_timestamps(duration: float, count: int, focus_end: bool = False) -> list[float]:
    if duration <= 0:
        return [0.0]
    if focus_end:
        start = max(0.0, duration - min(2.5, duration))
        end = max(start, duration - 0.08)
    else:
        start = 0.0
        end = max(0.0, duration - 0.08)
    return np.linspace(start, end, max(count, 1)).tolist()


def next_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"

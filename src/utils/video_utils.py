"""Basic video-file helpers for the initial pipeline scaffold.

This module intentionally does not extract PTS or perform frame-level
processing yet. If available, a later stage can call the system ``ffprobe``
executable directly rather than relying on a Python ffmpeg wrapper.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Union

try:  # Supports both ``import src.utils.video_utils`` and PYTHONPATH=src.
    from ..config import DATA_ROOT
except ImportError:  # pragma: no cover - exercised by the alternate import style.
    from config import DATA_ROOT


PathLike = Union[str, Path]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_video_files(root: Optional[PathLike] = None) -> List[Path]:
    """Return video files below *root*, sorted for deterministic processing."""

    search_root = Path(root) if root is not None else DATA_ROOT
    if not search_root.exists():
        return []
    if not search_root.is_dir():
        raise NotADirectoryError("Video search root is not a directory: " f"{search_root}")

    return sorted(
        (
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def get_ffprobe_path() -> Optional[str]:
    """Return the system ``ffprobe`` path when it is available."""

    return shutil.which("ffprobe")


def validate_video_path(video_path: PathLike) -> Path:
    """Validate and return an existing video path without reading its frames."""

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError("Video file does not exist: " f"{path}")
    if not path.is_file():
        raise IsADirectoryError("Video path is not a file: " f"{path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video extension: " f"{path.suffix}")
    return path


__all__ = [
    "VIDEO_EXTENSIONS",
    "find_video_files",
    "get_ffprobe_path",
    "validate_video_path",
]

"""Build a raw-data manifest for Problem 1.

The script is intentionally an audit step. It reads the source Excel workbook
and MP4 files, but never modifies, moves, renames, or converts them.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import pandas as pd

try:  # Direct execution: ``python src/01_build_manifest.py``.
    from config import (
        DATA_ROOT,
        FIGURE_DIR,
        LABEL_PATH,
        LOG_DIR,
        METADATA_DIR,
        PROJECT_ROOT,
    )
    from utils.io_utils import ensure_directory, normalize_id, safe_float
    from utils.video_utils import find_video_files, get_ffprobe_path
except ImportError:  # Module execution: ``python -m src.01_build_manifest``.
    from src.config import (
        DATA_ROOT,
        FIGURE_DIR,
        LABEL_PATH,
        LOG_DIR,
        METADATA_DIR,
        PROJECT_ROOT,
    )
    from src.utils.io_utils import ensure_directory, normalize_id, safe_float
    from src.utils.video_utils import find_video_files, get_ffprobe_path


MANIFEST_PATH = METADATA_DIR / "raw_manifest.csv"
SUMMARY_PATH = LOG_DIR / "manifest_summary.txt"

# The local project does not contain a separate copy of the problem statement.
# This is therefore a conservative, editable audit range for the current raw
# clips. It is only used to create duration_in_expected_range; it never removes
# or filters a sample. Replace these two values if the official statement gives
# a different explicit range.
EXPECTED_DURATION_MIN_SECONDS = 2.0
EXPECTED_DURATION_MAX_SECONDS = 40.0

# Used only to report a clearly visible audio/video duration discrepancy when
# both stream durations are available. It is not a deletion criterion.
AV_RELATIVE_ERROR_REPORT_THRESHOLD = 0.05

REQUIRED_LABEL_COLUMNS = ["video_id", "clip_id", "text", "label", "annotation"]
VIDEO_METADATA_COLUMNS = [
    "video_path",
    "file_name",
    "file_size_mb",
    "cv_readable",
    "fps",
    "frame_count",
    "width",
    "height",
    "cv_duration",
    "container_duration",
    "has_video",
    "video_codec",
    "video_width",
    "video_height",
    "avg_frame_rate",
    "video_duration",
    "video_time_base",
    "has_audio",
    "audio_codec",
    "audio_sample_rate",
    "audio_channels",
    "audio_duration",
    "audio_time_base",
]

LOGGER = logging.getLogger("build_manifest")


def configure_logging() -> None:
    """Configure concise console logging for the one-shot audit script."""

    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def is_missing(value: Any) -> bool:
    """Return whether a scalar should be treated as missing."""

    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def count_text_chars(value: Any) -> Optional[int]:
    """Count characters without changing the original text column."""

    if is_missing(value):
        return None
    return len(str(value))


def count_text_words(value: Any) -> Optional[int]:
    """Count whitespace-delimited words for the current basic audit only."""

    if is_missing(value):
        return None
    return len(str(value).split())


def make_sample_id(video_id: Any, clip_id: Any) -> Optional[str]:
    """Build the stable composite key from separately retained ID fields."""

    normalized_video_id = normalize_id(video_id)
    normalized_clip_id = normalize_id(clip_id)
    if normalized_video_id is None or normalized_clip_id is None:
        return None
    return f"{normalized_video_id}_{normalized_clip_id}"


def load_labels(label_path: Path) -> pd.DataFrame:
    """Read labels and add audit-only text and annotation-consistency fields."""

    labels = pd.read_excel(label_path, sheet_name=0, engine="openpyxl")
    missing_columns = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels]
    if missing_columns:
        raise ValueError(
            f"Label workbook is missing required columns: {', '.join(missing_columns)}"
        )

    # Keep the source text, label, and annotation values unchanged. ID values
    # are normalized only so Excel numeric cells and filename IDs can be joined.
    labels = labels[REQUIRED_LABEL_COLUMNS].copy()
    labels["video_id"] = labels["video_id"].map(normalize_id)
    labels["clip_id"] = labels["clip_id"].map(normalize_id)
    labels["sample_id"] = [
        make_sample_id(video_id, clip_id)
        for video_id, clip_id in zip(labels["video_id"], labels["clip_id"])
    ]
    labels["char_count"] = labels["text"].map(count_text_chars)
    labels["word_count"] = labels["text"].map(count_text_words)
    labels["annotation_expected"] = labels["label"].map(expected_annotation)
    labels["label_annotation_match"] = [
        annotation_match(label, annotation)
        for label, annotation in zip(labels["label"], labels["annotation"])
    ]
    labels["_video_id_key"] = labels["video_id"]
    labels["_clip_id_key"] = labels["clip_id"]
    return labels


def expected_annotation(label: Any) -> Optional[str]:
    """Map numeric sentiment to the expected annotation without changing it."""

    numeric_label = safe_float(label)
    if numeric_label is None:
        return None
    if numeric_label < 0:
        return "Negative"
    if numeric_label == 0:
        return "Neutral"
    return "Positive"


def annotation_match(label: Any, annotation: Any) -> Optional[bool]:
    """Compare source annotation to the expected label logic for auditing."""

    expected = expected_annotation(label)
    if expected is None:
        return None
    if is_missing(annotation):
        return False
    # Strip only for comparison; the original annotation value remains intact.
    return str(annotation).strip() == expected


def relative_video_path(path: Path) -> str:
    """Store a portable project-relative path in the manifest."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def recover_ids_from_path(video_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Recover IDs from the observed ``data1/video_id/clip_id.mp4`` layout."""

    try:
        relative_path = video_path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        return None, None

    if len(relative_path.parts) < 2:
        return None, normalize_id(video_path.stem)

    video_id = normalize_id(relative_path.parts[-2])
    clip_id = normalize_id(video_path.stem)
    return video_id, clip_id


def empty_opencv_result() -> Dict[str, Any]:
    """Create missing-value defaults for per-video OpenCV fields."""

    return {
        "cv_readable": None,
        "fps": None,
        "frame_count": None,
        "width": None,
        "height": None,
        "cv_duration": None,
    }


def inspect_with_opencv(video_path: Path) -> Dict[str, Any]:
    """Read basic container properties with OpenCV, isolating one-file errors."""

    result = empty_opencv_result()
    capture = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        result["cv_readable"] = bool(capture.isOpened())
        if not result["cv_readable"]:
            return result

        fps = safe_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = safe_float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_value = safe_float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = safe_float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result["fps"] = fps
        result["frame_count"] = (
            int(round(frame_count_value))
            if frame_count_value is not None and frame_count_value >= 0
            else None
        )
        result["width"] = (
            int(round(width_value)) if width_value is not None and width_value >= 0 else None
        )
        result["height"] = (
            int(round(height_value))
            if height_value is not None and height_value >= 0
            else None
        )

        # This is only a basic OpenCV check value. It must not be treated as
        # the sole source of truth for the strict physical timeline later.
        if result["frame_count"] is not None and fps is not None and fps > 0:
            result["cv_duration"] = result["frame_count"] / fps
    except Exception:
        LOGGER.exception("OpenCV inspection failed for %s", video_path)
    finally:
        if capture is not None:
            capture.release()
    return result


def parse_fraction(value: Any) -> Optional[float]:
    """Parse ffprobe rates such as ``30000/1001`` into a float."""

    if is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA"}:
        return None
    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        numerator = safe_float(numerator_text)
        denominator = safe_float(denominator_text)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    return safe_float(text)


def empty_ffprobe_result() -> Dict[str, Any]:
    """Create missing-value defaults for all requested ffprobe fields."""

    return {
        "container_duration": None,
        "has_video": None,
        "video_codec": None,
        "video_width": None,
        "video_height": None,
        "avg_frame_rate": None,
        "video_duration": None,
        "video_time_base": None,
        "has_audio": None,
        "audio_codec": None,
        "audio_sample_rate": None,
        "audio_channels": None,
        "audio_duration": None,
        "audio_time_base": None,
    }


def stream_duration(stream: Dict[str, Any]) -> Optional[float]:
    """Read a stream duration as a finite float when available."""

    return safe_float(stream.get("duration"))


def probe_with_ffprobe(video_path: Path, ffprobe_path: Optional[str]) -> Dict[str, Any]:
    """Run system ffprobe and extract stream/format metadata defensively."""

    result = empty_ffprobe_result()
    if ffprobe_path is None:
        return result

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if completed.returncode != 0:
            LOGGER.warning("ffprobe failed for %s: %s", video_path, completed.stderr.strip())
            return result

        payload = json.loads(completed.stdout)
        result["container_duration"] = safe_float(
            payload.get("format", {}).get("duration")
        )
        streams = payload.get("streams", [])
        video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]

        result["has_video"] = bool(video_streams)
        result["has_audio"] = bool(audio_streams)

        if video_streams:
            stream = video_streams[0]
            result["video_codec"] = stream.get("codec_name")
            result["video_width"] = stream.get("width")
            result["video_height"] = stream.get("height")
            result["avg_frame_rate"] = parse_fraction(stream.get("avg_frame_rate"))
            result["video_duration"] = stream_duration(stream)
            result["video_time_base"] = stream.get("time_base")

        if audio_streams:
            stream = audio_streams[0]
            result["audio_codec"] = stream.get("codec_name")
            result["audio_sample_rate"] = safe_float(stream.get("sample_rate"))
            result["audio_channels"] = stream.get("channels")
            result["audio_duration"] = stream_duration(stream)
            result["audio_time_base"] = stream.get("time_base")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError, ValueError):
        LOGGER.exception("ffprobe inspection failed for %s", video_path)
    return result


def inspect_video(video_path: Path, ffprobe_path: Optional[str]) -> Dict[str, Any]:
    """Collect one video record while isolating failures to that video."""

    video_id, clip_id = recover_ids_from_path(video_path)
    try:
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
    except OSError:
        file_size_mb = None

    record: Dict[str, Any] = {
        "video_id": video_id,
        "clip_id": clip_id,
        "sample_id": make_sample_id(video_id, clip_id),
        "video_path": relative_video_path(video_path),
        "file_name": video_path.name,
        "file_size_mb": file_size_mb,
    }
    record.update(inspect_with_opencv(video_path))
    record.update(probe_with_ffprobe(video_path, ffprobe_path))
    record["_video_id_key"] = video_id
    record["_clip_id_key"] = clip_id
    return record


def load_video_records(video_paths: Sequence[Path], ffprobe_path: Optional[str]) -> pd.DataFrame:
    """Scan all MP4s and preserve a row even if one file cannot be inspected."""

    records = [inspect_video(video_path, ffprobe_path) for video_path in video_paths]
    return pd.DataFrame(records)


def duplicate_pairs(frame: pd.DataFrame) -> Tuple[int, List[str]]:
    """Return duplicate-row count and duplicate composite IDs."""

    if frame.empty:
        return 0, []
    keys = ["_video_id_key", "_clip_id_key"]
    duplicate_mask = frame.duplicated(keys, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = sorted(
        frame.loc[duplicate_mask, "sample_id"].dropna().astype(str).unique().tolist()
    )
    return duplicate_rows, duplicate_ids


def choose_duration(row: pd.Series) -> Optional[float]:
    """Prefer ffprobe video duration and fall back to the OpenCV check value."""

    return safe_float(row.get("video_duration")) or safe_float(row.get("cv_duration"))


def duration_in_expected_range(value: Any) -> Optional[bool]:
    """Return an audit flag without using it to filter records."""

    duration = safe_float(value)
    if duration is None:
        return None
    return EXPECTED_DURATION_MIN_SECONDS <= duration <= EXPECTED_DURATION_MAX_SECONDS


def add_duration_audit_fields(manifest: pd.DataFrame) -> None:
    """Add AV duration checks and the configured range audit flag in place."""

    diffs: List[Optional[float]] = []
    relative_errors: List[Optional[float]] = []
    expected_flags: List[Optional[bool]] = []

    for _, row in manifest.iterrows():
        audio_duration = safe_float(row.get("audio_duration"))
        video_duration = safe_float(row.get("video_duration"))
        if audio_duration is None or video_duration is None:
            diffs.append(None)
            relative_errors.append(None)
        else:
            difference = abs(audio_duration - video_duration)
            denominator = max(audio_duration, video_duration)
            diffs.append(difference)
            relative_errors.append(difference / denominator if denominator > 0 else None)
        expected_flags.append(duration_in_expected_range(choose_duration(row)))

    manifest["av_duration_diff"] = diffs
    manifest["av_duration_relative_error"] = relative_errors
    manifest["duration_in_expected_range"] = expected_flags


def add_quality_flags(
    manifest: pd.DataFrame,
    label_duplicate_ids: Iterable[str],
    video_duplicate_ids: Iterable[str],
) -> None:
    """Add quality flags; flags audit data and never delete a sample.

    Definitions:
    - q_id: both IDs/sample_id exist and the sample ID is not duplicated in
      either source.
    - q_video: OpenCV opened the file successfully.
    - q_audio: ffprobe positively confirmed an audio stream; unknown metadata
      does not pass this flag.
    - q_label: the sample matched a label row and its annotation agrees with
      the requested label logic.
    - q_duration: a usable duration falls in the configured audit range.
    - basic_quality_pass: logical AND of all five flags. This is an audit
      result only; it is never used to remove samples.
    """

    label_duplicate_ids = set(label_duplicate_ids)
    video_duplicate_ids = set(video_duplicate_ids)
    q_id = []
    q_video = []
    q_audio = []
    q_label = []
    q_duration = []

    for _, row in manifest.iterrows():
        sample_id = row.get("sample_id")
        sample_id_text = None if is_missing(sample_id) else str(sample_id)
        q_id.append(
            bool(
                sample_id_text
                and not is_missing(row.get("video_id"))
                and not is_missing(row.get("clip_id"))
                and sample_id_text not in label_duplicate_ids
                and sample_id_text not in video_duplicate_ids
            )
        )
        q_video.append(row.get("cv_readable") is True)
        q_audio.append(row.get("has_audio") is True)
        q_label.append(
            row.get("source_match_status") == "both"
            and row.get("label_annotation_match") is True
        )
        q_duration.append(row.get("duration_in_expected_range") is True)

    manifest["q_id"] = q_id
    manifest["q_video"] = q_video
    manifest["q_audio"] = q_audio
    manifest["q_label"] = q_label
    manifest["q_duration"] = q_duration
    manifest["basic_quality_pass"] = [
        all(flags)
        for flags in zip(q_id, q_video, q_audio, q_label, q_duration)
    ]


def merge_labels_and_videos(labels: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge by video_id + clip_id and retain unmatched records."""

    label_columns = [
        "_video_id_key",
        "_clip_id_key",
        "video_id",
        "clip_id",
        "sample_id",
        "text",
        "label",
        "annotation",
        "char_count",
        "word_count",
        "annotation_expected",
        "label_annotation_match",
    ]
    video_columns = ["_video_id_key", "_clip_id_key"] + VIDEO_METADATA_COLUMNS
    merged = labels[label_columns].merge(
        videos[video_columns],
        on=["_video_id_key", "_clip_id_key"],
        how="outer",
        indicator=True,
        sort=False,
    )
    merged["source_match_status"] = merged["_merge"].map(
        {"both": "both", "left_only": "label_only", "right_only": "video_only"}
    )
    merged["video_id"] = merged["_video_id_key"]
    merged["clip_id"] = merged["_clip_id_key"]
    merged["sample_id"] = [
        make_sample_id(video_id, clip_id)
        for video_id, clip_id in zip(merged["video_id"], merged["clip_id"])
    ]
    return merged.drop(columns=["_merge"])


def format_count(value: Any) -> str:
    """Format a count-like value for the text summary."""

    return str(int(value))


def duplicate_summary(name: str, duplicate_rows: int, duplicate_ids: Sequence[str]) -> str:
    """Render duplicate-ID audit details."""

    ids = ", ".join(duplicate_ids) if duplicate_ids else "无"
    return f"{name}重复行数: {duplicate_rows}\n{name}重复sample_id: {ids}"


def abnormal_reasons(row: pd.Series) -> List[str]:
    """Collect non-destructive audit reasons for one manifest row."""

    reasons: List[str] = []
    status = row.get("source_match_status")
    if status != "both":
        reasons.append(status or "missing_match_status")
    if row.get("cv_readable") is not True:
        reasons.append("cv_unreadable")
    if row.get("has_audio") is False:
        reasons.append("missing_audio")
    if row.get("label_annotation_match") is False:
        reasons.append("label_annotation_mismatch")
    if row.get("duration_in_expected_range") is False:
        reasons.append("duration_out_of_expected_range")
    relative_error = safe_float(row.get("av_duration_relative_error"))
    if relative_error is not None and relative_error > AV_RELATIVE_ERROR_REPORT_THRESHOLD:
        reasons.append("av_duration_relative_error_high")
    return reasons


def duration_distribution(manifest: pd.DataFrame) -> str:
    """Render a compact distribution for the summary log."""

    durations = pd.to_numeric(manifest["cv_duration"], errors="coerce").dropna()
    if durations.empty:
        return "可用 cv_duration: 0"
    quantiles = durations.quantile([0.25, 0.5, 0.75])
    return (
        f"可用 cv_duration: {len(durations)}, "
        f"min={durations.min():.3f}, "
        f"q25={quantiles.loc[0.25]:.3f}, "
        f"median={quantiles.loc[0.5]:.3f}, "
        f"q75={quantiles.loc[0.75]:.3f}, "
        f"max={durations.max():.3f} 秒"
    )


def write_summary(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    videos: pd.DataFrame,
    label_duplicate_rows: int,
    label_duplicate_ids: Sequence[str],
    video_duplicate_rows: int,
    video_duplicate_ids: Sequence[str],
    ffprobe_path: Optional[str],
) -> None:
    """Write the requested human-readable audit summary."""

    status_counts = Counter(manifest["source_match_status"].dropna())
    cv_readable = manifest["cv_readable"]
    audio_present = int((manifest["has_audio"] == True).sum())
    audio_missing = int((manifest["has_audio"] == False).sum())
    audio_unknown = int(manifest["has_audio"].isna().sum())
    label_match_true = int((manifest["label_annotation_match"] == True).sum())
    label_match_false = int((manifest["label_annotation_match"] == False).sum())
    label_match_unknown = int(manifest["label_annotation_match"].isna().sum())

    abnormal: Dict[str, List[str]] = {}
    for _, row in manifest.iterrows():
        reasons = abnormal_reasons(row)
        if reasons:
            sample_id = row.get("sample_id")
            key = "<missing_sample_id>" if is_missing(sample_id) else str(sample_id)
            abnormal[key] = reasons

    lines = [
        "Problem 1 raw manifest summary",
        "================================",
        f"manifest 行数（外连接后）: {len(manifest)}",
        f"标签记录数: {len(labels)}",
        f"视频记录数: {len(videos)}",
        f"匹配记录数: {status_counts.get('both', 0)}",
        f"label_only 数: {status_counts.get('label_only', 0)}",
        f"video_only 数: {status_counts.get('video_only', 0)}",
        "",
        duplicate_summary("Excel", label_duplicate_rows, label_duplicate_ids),
        duplicate_summary("文件系统", video_duplicate_rows, video_duplicate_ids),
        "",
        f"OpenCV 可读取: {int((cv_readable == True).sum())}",
        f"OpenCV 不可读取: {int((cv_readable == False).sum())}",
        f"OpenCV 状态未知: {int(cv_readable.isna().sum())}",
        f"音频存在: {audio_present}",
        f"音频缺失: {audio_missing}",
        f"音频状态未知（通常表示 ffprobe 不可用或探测失败）: {audio_unknown}",
        f"ffprobe: {ffprobe_path or '未找到，已跳过 ffprobe 元数据'}",
        "",
        f"标签逻辑一致: {label_match_true}",
        f"标签逻辑不一致: {label_match_false}",
        f"标签逻辑未知: {label_match_unknown}",
        duration_distribution(manifest),
        f"配置的期望时长审计范围: {EXPECTED_DURATION_MIN_SECONDS:.1f}–{EXPECTED_DURATION_MAX_SECONDS:.1f} 秒",
        f"期望范围内: {int((manifest['duration_in_expected_range'] == True).sum())}",
        f"期望范围外: {int((manifest['duration_in_expected_range'] == False).sum())}",
        f"时长无法判断: {int(manifest['duration_in_expected_range'].isna().sum())}",
        "",
        "异常 sample_id:",
    ]
    if abnormal:
        lines.extend(f"- {sample_id}: {', '.join(reasons)}" for sample_id, reasons in abnormal.items())
    else:
        lines.append("- 无")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest() -> pd.DataFrame:
    """Run the complete raw-manifest audit and save its outputs."""

    configure_logging()
    ensure_directory(METADATA_DIR)
    ensure_directory(FIGURE_DIR)
    ensure_directory(LOG_DIR)

    if not LABEL_PATH.is_file():
        raise FileNotFoundError(f"Label workbook does not exist: {LABEL_PATH}")
    if not DATA_ROOT.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {DATA_ROOT}")

    labels = load_labels(LABEL_PATH)
    video_paths = find_video_files(DATA_ROOT)
    ffprobe_path = get_ffprobe_path()
    if ffprobe_path is None:
        LOGGER.warning(
            "系统未找到 ffprobe；将继续生成 manifest，但 ffprobe 字段和音频状态可能为空。"
        )
    else:
        LOGGER.info("使用系统 ffprobe: %s", ffprobe_path)

    LOGGER.info("读取标签记录: %d", len(labels))
    LOGGER.info("发现 MP4 文件: %d", len(video_paths))
    videos = load_video_records(video_paths, ffprobe_path)
    manifest = merge_labels_and_videos(labels, videos)

    label_duplicate_rows, label_duplicate_ids = duplicate_pairs(labels)
    video_duplicate_rows, video_duplicate_ids = duplicate_pairs(videos)

    add_duration_audit_fields(manifest)
    add_quality_flags(manifest, label_duplicate_ids, video_duplicate_ids)

    output_columns = [
        "video_id",
        "clip_id",
        "sample_id",
        "source_match_status",
        "text",
        "label",
        "annotation",
        "char_count",
        "word_count",
        "annotation_expected",
        "label_annotation_match",
    ] + VIDEO_METADATA_COLUMNS + [
        "av_duration_diff",
        "av_duration_relative_error",
        "duration_in_expected_range",
        "q_id",
        "q_video",
        "q_audio",
        "q_label",
        "q_duration",
        "basic_quality_pass",
    ]
    manifest = manifest[output_columns]
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    write_summary(
        manifest,
        labels,
        videos,
        label_duplicate_rows,
        label_duplicate_ids,
        video_duplicate_rows,
        video_duplicate_ids,
        ffprobe_path,
    )
    LOGGER.info("已写入 manifest: %s", MANIFEST_PATH)
    LOGGER.info("已写入汇总: %s", SUMMARY_PATH)
    return manifest


def main() -> None:
    """CLI entry point."""

    build_manifest()


if __name__ == "__main__":
    main()

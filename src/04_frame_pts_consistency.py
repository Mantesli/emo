"""Compare OpenCV metadata and decoded frame counts with the saved video PTS.

This is a read-only diagnostic step. It does not alter timestamp NPZ files or
derive a replacement timeline from FPS.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

try:  # Direct execution: python src/04_frame_pts_consistency.py
    from config import METADATA_DIR, OUTPUT_ROOT, PROJECT_ROOT
    from utils.io_utils import ensure_directory, safe_float
except ImportError:  # Module execution: python -m src.04_frame_pts_consistency
    from src.config import METADATA_DIR, OUTPUT_ROOT, PROJECT_ROOT
    from src.utils.io_utils import ensure_directory, safe_float


MANIFEST_PATH = METADATA_DIR / "raw_manifest.csv"
TIMESTAMP_SUMMARY_PATH = METADATA_DIR / "timestamp_summary.csv"
TIMESTAMP_DIR = OUTPUT_ROOT / "timestamps"
OUTPUT_PATH = METADATA_DIR / "frame_pts_consistency.csv"
REPORT_PATH = OUTPUT_ROOT / "logs" / "frame_pts_consistency_report.txt"

FOCUS_SAMPLE_IDS = ("-s9qJ7ATP7w_6", "-9y-fZ3swSY_4", "-yRb-Jum7EQ_1")
MAX_FFPROBE_SAMPLES = 5


def resolve_manifest_path(value: Any) -> Path:
    """Resolve a manifest path while accepting paths written on this host."""
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def decode_with_opencv(video_path: Path) -> int:
    """Count frames actually returned by VideoCapture.read()."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open video")
    decoded_frame_count = 0
    try:
        while True:
            ret, _frame = capture.read()
            if not ret:
                break
            decoded_frame_count += 1
    finally:
        capture.release()
    return decoded_frame_count


def ffprobe_count_frames(ffprobe: str, video_path: Path) -> Dict[str, Any]:
    """Use ffprobe's decoder-backed nb_read_frames count for a second check."""
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames",
        "-of", "json",
        str(video_path),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = streams[0]
    read_count = stream.get("nb_read_frames")
    return {
        "ffprobe_count_frames": int(read_count) if read_count not in (None, "N/A") else None,
        "ffprobe_nb_frames": stream.get("nb_frames"),
    }


def load_pts_counts() -> Dict[str, int]:
    """Read saved video_pts arrays without changing their contents."""
    counts: Dict[str, int] = {}
    for path in TIMESTAMP_DIR.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            if "video_pts" not in payload:
                raise ValueError(f"{path} has no video_pts array")
            counts[path.stem] = int(len(payload["video_pts"]))
    return counts


def inspect_sample(
    row: pd.Series,
    pts_counts: Dict[str, int],
    irregularity_by_id: Dict[str, Any],
) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "metadata_frame_count": safe_float(row.get("frame_count")),
        "actual_decoded_frame_count": None,
        "pts_count": pts_counts.get(sample_id),
        "video_timestamp_irregularity": safe_float(irregularity_by_id.get(sample_id)),
        "video_path": str(row.get("video_path", "")),
        "status": "DECODE_ERROR",
        "decode_error": "",
    }
    video_path = resolve_manifest_path(row.get("video_path"))
    try:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        record["actual_decoded_frame_count"] = decode_with_opencv(video_path)
        if record["pts_count"] is None:
            raise FileNotFoundError(f"Missing timestamps/{sample_id}.npz or video_pts")
        metadata_count = record["metadata_frame_count"]
        decoded_count = record["actual_decoded_frame_count"]
        pts_count = record["pts_count"]
        record["metadata_vs_decode_diff"] = (
            metadata_count - decoded_count if metadata_count is not None else None
        )
        record["decode_vs_pts_diff"] = decoded_count - pts_count
        record["metadata_matches_decode"] = (
            metadata_count == decoded_count if metadata_count is not None else None
        )
        record["decode_matches_pts"] = decoded_count == pts_count
        record["decode_pts_ratio"] = decoded_count / pts_count if pts_count else None
        if decoded_count != pts_count:
            record["status"] = "DECODE_PTS_MISMATCH"
        elif metadata_count != decoded_count:
            record["status"] = "METADATA_ONLY_MISMATCH"
        else:
            record["status"] = "OK"
    except Exception as exc:
        record["decode_error"] = f"{type(exc).__name__}: {exc}"
    return record


def finite_abs_max(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().abs()
    return float(values.max()) if len(values) else None


def write_report(
    summary: pd.DataFrame,
    ffprobe_results: Dict[str, Dict[str, Any]],
    ffprobe_errors: Dict[str, str],
    ffprobe: Optional[str],
) -> None:
    metadata_match_count = int((summary["metadata_matches_decode"] == True).sum())
    decode_pts_match_count = int((summary["decode_matches_pts"] == True).sum())
    metadata_mismatch = summary[summary["metadata_matches_decode"] == False]
    decode_pts_mismatch = summary[summary["status"] == "DECODE_PTS_MISMATCH"]
    max_metadata_diff = finite_abs_max(summary["metadata_vs_decode_diff"])
    max_decode_pts_diff = finite_abs_max(summary["decode_vs_pts_diff"])

    lines = [
        "视频解码帧数与 PTS 数量一致性报告",
        "=" * 36,
        f"总样本数: {len(summary)}",
        f"ffprobe: {ffprobe or '未找到'}",
        "本阶段仅读取媒体并诊断计数；未修改任何 PTS、NPZ 或源视频，未使用 FPS 生成时间戳。",
        f"metadata_frame_count == actual_decoded_frame_count: {metadata_match_count}/{len(summary)}",
        f"actual_decoded_frame_count == pts_count: {decode_pts_match_count}/{len(summary)}",
        f"metadata 与实际解码不一致样本数: {len(metadata_mismatch)}",
        f"实际解码与 PTS 数量冲突样本数: {len(decode_pts_mismatch)}",
        f"DECODE_ERROR 样本数: {int((summary['status'] == 'DECODE_ERROR').sum())}",
        f"最大 |metadata_vs_decode_diff|: {max_metadata_diff if max_metadata_diff is not None else 'NA'}",
        f"最大 |decode_vs_pts_diff|: {max_decode_pts_diff if max_decode_pts_diff is not None else 'NA'}",
        "",
        "DECODE_PTS_MISMATCH 样本:",
    ]
    if decode_pts_mismatch.empty:
        lines.append("无")
    else:
        for _, row in decode_pts_mismatch.sort_values(
            "decode_vs_pts_diff", key=lambda values: values.abs(), ascending=False
        ).iterrows():
            sid = str(row["sample_id"])
            probe = ffprobe_results.get(sid, {})
            lines.append(
                f"{sid}: metadata={row['metadata_frame_count']}, "
                f"opencv_decode={row['actual_decoded_frame_count']}, pts={row['pts_count']}, "
                f"ffprobe_count_frames={probe.get('ffprobe_count_frames', 'NA')}, "
                f"ffprobe_nb_frames={probe.get('ffprobe_nb_frames', 'NA')}, "
                f"ffprobe_error={ffprobe_errors.get(sid, '无')}"
            )
    lines.extend(["", "二次 ffprobe -count_frames 复核（差异最大的最多 5 条）:"])
    if not ffprobe_results and not ffprobe_errors:
        lines.append("无样本需要复核")
    else:
        for sid in list(ffprobe_results) + [k for k in ffprobe_errors if k not in ffprobe_results]:
            probe = ffprobe_results.get(sid, {})
            lines.append(
                f"{sid}: opencv_decode={summary.loc[summary.sample_id == sid, 'actual_decoded_frame_count'].iloc[0]}, "
                f"ffprobe_count_frames={probe.get('ffprobe_count_frames', 'NA')}, "
                f"ffprobe_nb_frames={probe.get('ffprobe_nb_frames', 'NA')}, "
                f"pts={summary.loc[summary.sample_id == sid, 'pts_count'].iloc[0]}, "
                f"error={ffprobe_errors.get(sid, '无')}"
            )
    lines.extend(["", "重点样本:"])
    by_id = summary.set_index("sample_id", drop=False)
    for sid in FOCUS_SAMPLE_IDS:
        if sid not in by_id.index:
            lines.append(f"{sid}: 不在 raw_manifest.csv 中")
            continue
        row = by_id.loc[sid]
        lines.append(
            f"{sid}: metadata={row['metadata_frame_count']}, "
            f"opencv_decode={row['actual_decoded_frame_count']}, pts={row['pts_count']}, "
            f"irregularity={row['video_timestamp_irregularity']}, status={row['status']}"
        )
    controls = summary[
        (pd.to_numeric(summary["metadata_frame_count"], errors="coerce")
         == pd.to_numeric(summary["pts_count"], errors="coerce"))
        & (summary["status"] == "OK")
    ]
    if controls.empty:
        lines.append("Control 样本（metadata_frame_count == pts_count 且三者一致）: 未找到")
    else:
        row = controls.iloc[0]
        lines.append(
            f"Control 样本: {row['sample_id']} (metadata={row['metadata_frame_count']}, "
            f"opencv_decode={row['actual_decoded_frame_count']}, pts={row['pts_count']})"
        )
    lines.extend([
        "",
        "映射判断: 只有在每个样本 actual_decoded_frame_count == pts_count，且解码顺序对应 ffprobe 帧顺序的前提下，",
        "计数证据才支持 decoded_frame[k] -> video_pts[k] 的逐项映射；计数不一致样本必须保留 mismatch 并等待人工判断。",
        "计数相等是必要证据，不单独证明解码器在帧排序/丢帧策略上完全一致。",
    ])
    ensure_directory(REPORT_PATH.parent)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    if not TIMESTAMP_SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing timestamp summary: {TIMESTAMP_SUMMARY_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"sample_id": str, "video_path": str})
    ts_summary = pd.read_csv(TIMESTAMP_SUMMARY_PATH, dtype={"sample_id": str})
    required = {"sample_id", "video_path", "frame_count"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"raw_manifest.csv is missing columns: {', '.join(sorted(missing))}")
    manifest = manifest[manifest["sample_id"].notna() & manifest["video_path"].notna()]
    pts_counts = load_pts_counts()
    irregularity_by_id = (
        ts_summary.set_index("sample_id")["video_timestamp_irregularity"].to_dict()
        if {"sample_id", "video_timestamp_irregularity"}.issubset(ts_summary.columns)
        else {}
    )
    records: List[Dict[str, Any]] = []
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        print(f"[{index}/{len(manifest)}] {row['sample_id']}", flush=True)
        records.append(inspect_sample(row, pts_counts, irregularity_by_id))
    summary = pd.DataFrame(records)
    mismatch = summary[summary["status"] == "DECODE_PTS_MISMATCH"].copy()
    mismatch["_abs_diff"] = pd.to_numeric(mismatch["decode_vs_pts_diff"], errors="coerce").abs()
    selected = mismatch.sort_values("_abs_diff", ascending=False).head(MAX_FFPROBE_SAMPLES)
    ffprobe = shutil.which("ffprobe")
    probe_results: Dict[str, Dict[str, Any]] = {}
    probe_errors: Dict[str, str] = {}
    if ffprobe:
        for _, row in selected.iterrows():
            sid = str(row["sample_id"])
            video_path = resolve_manifest_path(row["video_path"])
            try:
                probe_results[sid] = ffprobe_count_frames(ffprobe, video_path)
            except Exception as exc:
                probe_errors[sid] = f"{type(exc).__name__}: {exc}"
    elif not selected.empty:
        probe_errors = {str(row["sample_id"]): "ffprobe not found on PATH" for _, row in selected.iterrows()}

    summary["ffprobe_count_frames"] = summary["sample_id"].map(
        {sid: item.get("ffprobe_count_frames") for sid, item in probe_results.items()}
    )
    summary["ffprobe_nb_frames"] = summary["sample_id"].map(
        {sid: item.get("ffprobe_nb_frames") for sid, item in probe_results.items()}
    )
    summary["ffprobe_error"] = summary["sample_id"].map(probe_errors).fillna("")
    columns = [
        "sample_id", "metadata_frame_count", "actual_decoded_frame_count", "pts_count",
        "metadata_vs_decode_diff", "decode_vs_pts_diff", "metadata_matches_decode",
        "decode_matches_pts", "decode_pts_ratio", "video_timestamp_irregularity", "status",
        "ffprobe_count_frames", "ffprobe_nb_frames", "ffprobe_error", "decode_error", "video_path",
    ]
    for column in columns:
        if column not in summary:
            summary[column] = None
    ensure_directory(OUTPUT_PATH.parent)
    summary[columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    write_report(summary, probe_results, probe_errors, ffprobe)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return summary


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

"""Audit physical audio/video timelines from MP4 presentation timestamps.

This step deliberately extracts no emotion, lexical, facial, or acoustic
features. FPS metadata and frame_count / fps are descriptive estimates only;
the alignment timeline saved here is built from ffprobe frame PTS values.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Direct execution: python src/03_timestamp_audit.py
    from config import METADATA_DIR, OUTPUT_ROOT, PROJECT_ROOT
    from utils.io_utils import ensure_directory, safe_float
except ImportError:  # Module execution: python -m src.03_timestamp_audit
    from src.config import METADATA_DIR, OUTPUT_ROOT, PROJECT_ROOT
    from src.utils.io_utils import ensure_directory, safe_float


MANIFEST_PATH = METADATA_DIR / "raw_manifest.csv"
SUMMARY_PATH = METADATA_DIR / "timestamp_summary.csv"
REPORT_PATH = METADATA_DIR / "timestamp_audit_report.txt"
TIMESTAMP_DIR = OUTPUT_ROOT / "timestamps"
FIGURE_DIR = OUTPUT_ROOT / "figures" / "timestamp"
IRREGULARITY_THRESHOLD = 0.01


def run_ffprobe(ffprobe: str, video_path: Path, *args: str) -> Dict[str, Any]:
    command = [ffprobe, "-v", "error", *args, "-of", "json", str(video_path)]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=180,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
    return json.loads(result.stdout)


def ffprobe_json(ffprobe: str, video_path: Path) -> Dict[str, Any]:
    """Read metadata, decoded video frames, and audio packets separately.

    ffprobe combines -show_frames and -show_packets into a single
    ``packets_and_frames`` collection on some builds, losing the distinction;
    separate stream-selected calls keep the two timestamp domains explicit.
    """
    streams = run_ffprobe(
        ffprobe, video_path, "-show_streams", "-show_entries",
        "stream=index,codec_type,start_time,duration,time_base,sample_rate",
    ).get("streams", [])
    frames = run_ffprobe(
        ffprobe, video_path, "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pts_time",
    ).get("frames", [])
    packets = run_ffprobe(
        ffprobe, video_path, "-select_streams", "a:0", "-show_packets",
        "-show_entries",
        "packet=stream_index,pts_time,duration_time:"
        "packet_side_data=side_data_type,skip_samples,discard_padding",
    ).get("packets", [])
    return {"streams": streams, "frames": frames, "packets": packets}


def first_valid(*values: Any) -> Optional[float]:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def process_sample(row: pd.Series, ffprobe: str) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    video_path = (PROJECT_ROOT / str(row["video_path"])).resolve()
    record: Dict[str, Any] = {"sample_id": sample_id, "video_path": str(video_path)}
    try:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        probe = ffprobe_json(ffprobe, video_path)
        streams = probe.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        vstream = video_streams[0] if video_streams else {}
        astream = audio_streams[0] if audio_streams else {}

        # best_effort_timestamp_time is preferred; pts_time is the fallback.
        # ``-select_streams v:0`` restricts -show_frames to video already;
        # some ffprobe builds omit frame.media_type from JSON entirely.
        video_frames = probe.get("frames", [])
        video_pts_list = [
            first_valid(frame.get("best_effort_timestamp_time"), frame.get("pts_time"))
            for frame in video_frames
        ]
        valid_pts = [(i, pts) for i, pts in enumerate(video_pts_list) if pts is not None]
        frame_indices = np.asarray([i for i, _ in valid_pts], dtype=np.int64)
        video_pts = np.asarray([pts for _, pts in valid_pts], dtype=np.float64)
        intervals = np.diff(video_pts)
        positive_intervals = intervals[intervals > 0]
        mean_interval = float(np.mean(intervals)) if len(intervals) else math.nan
        std_interval = float(np.std(intervals)) if len(intervals) else math.nan
        irregularity = (
            std_interval / mean_interval
            if len(intervals) and mean_interval > 0 else math.nan
        )
        monotonic = bool(len(intervals) == 0 or np.all(intervals > 0))
        # CFR-like is judged from observed PTS intervals, allowing timestamp
        # quantization; nominal FPS metadata is not used for this decision.
        near_constant = bool(
            len(intervals) > 0 and mean_interval > 0
            and np.all(np.abs(intervals - mean_interval) <= max(mean_interval * 0.01, 1e-9))
        )

        audio_index = astream.get("index")
        packets = [
            p for p in probe.get("packets", [])
            if audio_index is not None and str(p.get("stream_index")) == str(audio_index)
        ]
        audio_packet_pts = [safe_float(p.get("pts_time")) for p in packets]
        audio_packet_pts = [t for t in audio_packet_pts if t is not None]
        audio_start = first_valid(astream.get("start_time"))
        audio_duration = safe_float(astream.get("duration"))
        first_audio_pts = min(audio_packet_pts) if audio_packet_pts else audio_start
        last_audio_pts = max(audio_packet_pts) if audio_packet_pts else None
        # MP4/AAC packets may carry encoder priming at negative PTS. Respect
        # ffprobe's Skip Samples side data when estimating the first audible
        # audio time; retain the raw packet PTS separately for full provenance.
        sample_rate = safe_float(astream.get("sample_rate"))
        first_packet = next((p for p in packets if safe_float(p.get("pts_time")) is not None), {})
        first_side = first_packet.get("side_data_list", [])
        first_skip = next((safe_float(sd.get("skip_samples"), 0.0) for sd in first_side if sd.get("side_data_type") == "Skip Samples"), 0.0)
        first_audio_time = (
            first_audio_pts + first_skip / sample_rate
            if first_audio_pts is not None and sample_rate and first_skip else first_audio_pts
        )
        # For the end comparison, include final packet duration and subtract
        # any encoder discard padding reported on that packet.
        final_packet_end = None
        final_discard = 0.0
        if packets and audio_packet_pts:
            for packet in reversed(packets):
                packet_pts = safe_float(packet.get("pts_time"))
                if packet_pts is not None:
                    packet_duration = safe_float(packet.get("duration_time"))
                    side_data = packet.get("side_data_list", [])
                    final_discard = next((safe_float(sd.get("discard_padding"), 0.0) for sd in side_data if sd.get("side_data_type") == "Skip Samples"), 0.0)
                    final_packet_end = packet_pts + (packet_duration or 0.0)
                    if sample_rate and final_discard:
                        final_packet_end -= final_discard / sample_rate
                    break
        video_start = float(video_pts[0]) if len(video_pts) else first_valid(vstream.get("start_time"))
        video_last = float(video_pts[-1]) if len(video_pts) else None
        sample_start = min(t for t in (video_start, first_audio_time) if t is not None) if any(
            t is not None for t in (video_start, first_audio_time)
        ) else None
        relative_pts = video_pts - sample_start if sample_start is not None else np.full(video_pts.shape, np.nan)

        record.update({
            "first_video_pts": video_start,
            "last_video_pts": video_last,
            "number_of_video_pts": int(len(video_pts)),
            "mean_frame_interval": mean_interval,
            "std_frame_interval": std_interval,
            "min_frame_interval": float(np.min(intervals)) if len(intervals) else math.nan,
            "max_frame_interval": float(np.max(intervals)) if len(intervals) else math.nan,
            "video_timestamp_irregularity": irregularity,
            "video_pts_monotonic": monotonic if len(video_pts) else None,
            "frame_interval_near_constant": near_constant,
            "nonpositive_frame_interval_count": int(np.sum(intervals <= 0)),
            "video_start_time": safe_float(vstream.get("start_time")),
            "video_time_base": vstream.get("time_base"),
            "audio_start_time": audio_start,
            "audio_duration": audio_duration,
            "audio_time_base": astream.get("time_base"),
            "audio_sample_rate": sample_rate,
            "first_audio_pts": first_audio_pts,
            "last_audio_pts": last_audio_pts,
            "first_audio_time": first_audio_time,
            "audio_initial_skip_samples": int(first_skip or 0),
            "audio_final_discard_padding": int(final_discard or 0),
            "last_audio_packet_end": final_packet_end,
            "audio_pts_source": "packet_pts_time" if audio_packet_pts else "stream_start_time_fallback",
            "audio_packet_count": len(audio_packet_pts),
            # Positive means audio starts/ends later than video; negative means earlier.
            "av_start_offset": first_audio_time - video_start if first_audio_time is not None and video_start is not None else math.nan,
            # End uses last audio packet end (PTS + duration) vs last video frame PTS.
            "av_end_offset": final_packet_end - video_last if final_packet_end is not None and video_last is not None else math.nan,
            "sample_start_time": sample_start,
            "timestamp_status": "ok" if len(video_pts) else "no_video_pts",
        })

        np.savez_compressed(
            TIMESTAMP_DIR / f"{sample_id}.npz",
            frame_index=frame_indices,
            video_pts=video_pts,
            video_relative_pts=relative_pts,
            frame_interval=intervals,
        )
        record["_frame_indices"] = frame_indices
        record["_video_pts"] = video_pts
        record["_intervals"] = intervals
    except Exception as exc:
        record.update({
            "timestamp_status": f"error: {type(exc).__name__}: {exc}",
            "number_of_video_pts": 0,
            "video_pts_monotonic": None,
            "frame_interval_near_constant": None,
        })
    return record


def make_plot(record: Dict[str, Any], label: str) -> None:
    indices = record.get("_frame_indices", np.asarray([], dtype=int))
    pts = record.get("_video_pts", np.asarray([], dtype=float))
    intervals = record.get("_intervals", np.asarray([], dtype=float))
    if not len(pts):
        return
    sample_id = record["sample_id"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(indices, pts, linewidth=0.8)
    axes[0].set(title=f"{sample_id} ({label}): frame index vs physical PTS", xlabel="Frame index", ylabel="PTS (s)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(indices[1:], intervals, linewidth=0.8)
    axes[1].set(title="Observed frame interval vs frame index", xlabel="Frame index", ylabel="Interval (s)")
    axes[1].grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / f"{sample_id}_{label}.png", dpi=160)
    plt.close(fig)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(summary: pd.DataFrame, all_records: Sequence[Dict[str, Any]], ffprobe: str) -> None:
    valid = summary[summary["number_of_video_pts"].fillna(0) > 0]
    monotonic_count = int((valid["video_pts_monotonic"] == True).sum())
    nonmonotonic = summary.loc[summary["video_pts_monotonic"] == False, "sample_id"].astype(str).tolist()
    missing = summary.loc[summary["number_of_video_pts"].fillna(0) == 0, "sample_id"].astype(str).tolist()
    irregular = summary.loc[
        pd.to_numeric(summary["video_timestamp_irregularity"], errors="coerce") > IRREGULARITY_THRESHOLD,
        "sample_id",
    ].astype(str).tolist()
    near_count = int((valid["frame_interval_near_constant"] == True).sum())
    offsets = pd.to_numeric(summary["av_start_offset"], errors="coerce").dropna()
    late_count = int((offsets > 0).sum())
    early_count = int((offsets < 0).sum())
    median_duration = pd.to_numeric(valid["last_video_pts"] - valid["first_video_pts"], errors="coerce")
    ordinary_record = None
    if len(median_duration):
        ordinary_idx = (median_duration - median_duration.median()).abs().idxmin()
        ordinary_record = next((r for r in all_records if r["sample_id"] == summary.loc[ordinary_idx, "sample_id"]), None)
    longest_id = (median_duration.idxmax() if len(median_duration) else None)
    longest_record = next((r for r in all_records if longest_id is not None and r["sample_id"] == summary.loc[longest_id, "sample_id"]), None)
    if ordinary_record:
        make_plot(ordinary_record, "ordinary")
    if longest_record and (not ordinary_record or longest_record["sample_id"] != ordinary_record["sample_id"]):
        make_plot(longest_record, "longest")
    if irregular:
        anomaly_id = max(irregular, key=lambda sid: safe_float(summary.loc[summary.sample_id == sid, "video_timestamp_irregularity"].iloc[0], -1))
        anomaly_record = next((r for r in all_records if r["sample_id"] == anomaly_id), None)
        if anomaly_record and anomaly_id not in {ordinary_record and ordinary_record["sample_id"], longest_record and longest_record["sample_id"]}:
            make_plot(anomaly_record, "irregular")

    reports = [
        "音视频原始物理时间轴审计报告",
        "=" * 32,
        f"样本数: {len(summary)}；ffprobe: {ffprobe}",
        "本阶段仅读取媒体时间戳；未提取情感/文本/人脸/声学特征，未做 Forced Alignment。",
        "时间依据：FPS metadata 是名义帧率；frame_count / fps 只是估算时长；本审计后续对齐使用 ffprobe 的真实帧 PTS（优先 best_effort_timestamp_time，回退 pts_time）。",
        "统一相对时间：sample_start_time = 音视频各自首个可用时间戳的较小值；video_relative_pts = video_pts - sample_start_time，原始 PTS 保留不变。",
        "av_start_offset = first audible audio time - first video time；正值表示音频起点晚于视频，负值表示音频起点早于视频。若 AAC packet 有 Skip Samples priming 信息，会扣除被跳过的样本；first_audio_pts 仍保留原始 packet PTS。",
        "av_end_offset = 最后音频 packet 的 PTS+duration - 最后视频帧 PTS；正值表示音频尾端晚于最后视频帧时间戳。",
        "音频 start_time/duration/time_base 来自 stream metadata；本次额外读取 audio packet PTS，若缺失则回退 stream start_time。对 AAC packet 的 Skip Samples/Discard Padding 边界信息予以应用；否则 packet PTS 会把编码 priming 误当作可听起点。",
        "帧间隔近似恒定的判定基于观测 interval 与均值的偏差不超过 1%，不参考 FPS metadata。irregularity > 0.01 列为疑似异常/VFR 描述性筛查阈值，不等同于编码器级 VFR 证明。",
        "",
        f"视频 PTS 可读取: {len(valid)}/{len(summary)}；不可读取/失败: {', '.join(missing) if missing else '无'}",
        f"PTS 单调递增: {monotonic_count}/{len(valid)}；非单调 sample_id: {', '.join(nonmonotonic) if nonmonotonic else '无'}",
        f"帧间隔近似恒定: {near_count}/{len(valid)}；不满足近似恒定的有效样本: {', '.join(valid.loc[valid.frame_interval_near_constant != True, 'sample_id'].astype(str).tolist()) or '无'}",
        f"疑似 irregularity/VFR（irregularity > {IRREGULARITY_THRESHOLD:g}）: {', '.join(irregular) if irregular else '无'}",
        f"音频 packet PTS 可读样本: {int((summary.audio_pts_source == 'packet_pts_time').sum())}/{len(summary)}；采用 stream start_time 回退: {int((summary.audio_pts_source == 'stream_start_time_fallback').sum())}；起始偏移 audio later/earlier/equal: {late_count}/{early_count}/{int((offsets == 0).sum())}",
        f"起始偏移中位数（audio-video）: {fmt(offsets.median() if len(offsets) else math.nan)} s；范围: {fmt(offsets.min() if len(offsets) else math.nan)} 至 {fmt(offsets.max() if len(offsets) else math.nan)} s",
        "",
        "图像选择：普通样本取视频 PTS 时长最接近中位数者；最长样本取 PTS 时长最大者；如有疑似异常，再取 irregularity 最大者。",
        f"图像目录: {FIGURE_DIR}",
        f"NPZ 目录: {TIMESTAMP_DIR}",
        "",
        "限制：video last time 是最后一帧的呈现时间戳，不是解码显示时长终点；音频边界依据 packet PTS 与 packet duration，不是样本级 waveform 测量。容器/流时间基下的 PTS 为本阶段可得的物理时间组织证据。",
    ]
    REPORT_PATH.write_text("\n".join(reports) + "\n", encoding="utf-8")


def run() -> pd.DataFrame:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe was not found on PATH")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}; run src/01_build_manifest.py first")
    for directory in (TIMESTAMP_DIR, FIGURE_DIR, REPORT_PATH.parent):
        ensure_directory(directory)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"sample_id": str, "video_path": str})
    manifest = manifest[manifest["sample_id"].notna() & manifest["video_path"].notna()]
    records = []
    for i, (_, row) in enumerate(manifest.iterrows(), start=1):
        print(f"[{i}/{len(manifest)}] {row['sample_id']}", flush=True)
        records.append(process_sample(row, ffprobe))
    public_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    summary = pd.DataFrame(public_records)
    # Ensure the requested columns exist even for an entirely failed run.
    required = [
        "sample_id", "first_video_pts", "last_video_pts", "number_of_video_pts",
        "mean_frame_interval", "std_frame_interval", "min_frame_interval", "max_frame_interval",
        "video_timestamp_irregularity", "video_pts_monotonic", "frame_interval_near_constant",
        "audio_start_time", "audio_duration", "audio_time_base", "first_audio_pts", "last_audio_pts", "first_audio_time",
        "av_start_offset", "av_end_offset", "sample_start_time", "video_time_base", "timestamp_status",
    ]
    for column in required:
        if column not in summary:
            summary[column] = np.nan
    ordered = required + [c for c in summary.columns if c not in required]
    summary[ordered].to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    write_report(summary, records, ffprobe)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return summary


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

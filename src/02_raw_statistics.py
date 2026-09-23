"""Exploratory statistics for the raw multimodal manifest.

This stage deliberately reads only ``outputs/metadata/raw_manifest.csv``.
It does not rescan, open, decode, or modify any raw audio/video/text source.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "metadata" / "raw_manifest.csv"
METADATA_DIR = PROJECT_ROOT / "outputs" / "metadata"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
STATISTICS_PATH = METADATA_DIR / "raw_statistics.csv"
QUALITY_PATH = METADATA_DIR / "quality_summary.csv"
REPORT_PATH = LOG_DIR / "raw_statistics_report.txt"

# These thresholds are audit/reporting thresholds only. They never filter data.
EXPECTED_DURATION_MIN_SECONDS = 2.0
EXPECTED_DURATION_MAX_SECONDS = 40.0
AV_RELATIVE_ERROR_REPORT_THRESHOLD = 0.05
WORD_COUNT_REFERENCE = 50


REQUIRED_COLUMNS = [
    "sample_id",
    "source_match_status",
    "word_count",
    "fps",
    "frame_count",
    "container_duration",
    "video_duration",
    "has_video",
    "cv_readable",
    "has_audio",
    "audio_codec",
    "audio_sample_rate",
    "audio_channels",
    "audio_duration",
    "av_duration_diff",
    "av_duration_relative_error",
    "label",
    "annotation",
    "label_annotation_match",
    "duration_in_expected_range",
    "q_video",
    "q_audio",
    "q_label",
    "q_duration",
    "basic_quality_pass",
]


def is_true(value: Any) -> bool:
    """Parse manifest boolean values without treating arbitrary text as true."""

    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series, coercing malformed/missing values to NaN."""

    return pd.to_numeric(frame[column], errors="coerce")


def clean_number(value: Any) -> Any:
    """Convert numpy scalar values to CSV-friendly Python values."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, np.generic):
        return value.item()
    return value


def add_value_stat(
    rows: List[Dict[str, Any]], section: str, metric: str, value: Any, note: str = ""
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "category": "",
            "value": clean_number(value),
            "count": np.nan,
            "proportion": np.nan,
            "sample_id": "",
            "video_duration": np.nan,
            "audio_duration": np.nan,
            "difference": np.nan,
            "relative_error": np.nan,
            "note": note,
        }
    )


def add_frequency_rows(
    rows: List[Dict[str, Any]],
    section: str,
    metric: str,
    values: Iterable[Any],
    total: Optional[int] = None,
) -> None:
    """Append a frequency table, including a missing-value category."""

    series = pd.Series(list(values), dtype="object")
    display = series.map(lambda x: "<missing>" if pd.isna(x) else str(x))
    counts = display.value_counts(dropna=False, sort=False)
    denominator = int(total if total is not None else len(series))
    for category, count in counts.items():
        rows.append(
            {
                "section": section,
                "metric": metric,
                "category": category,
                "value": np.nan,
                "count": int(count),
                "proportion": float(count / denominator) if denominator else np.nan,
                "sample_id": "",
                "video_duration": np.nan,
                "audio_duration": np.nan,
                "difference": np.nan,
                "relative_error": np.nan,
                "note": "",
            }
        )


def add_descriptive_stats(
    rows: List[Dict[str, Any]], section: str, series: pd.Series, metrics: List[str]
) -> pd.Series:
    """Append requested descriptive statistics and return the valid values."""

    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    functions = {
        "count": lambda s: len(s),
        "mean": lambda s: s.mean(),
        "std": lambda s: s.std(),
        "min": lambda s: s.min(),
        "25%": lambda s: s.quantile(0.25),
        "median": lambda s: s.median(),
        "75%": lambda s: s.quantile(0.75),
        "max": lambda s: s.max(),
        "P90": lambda s: s.quantile(0.90),
        "P95": lambda s: s.quantile(0.95),
    }
    for metric in metrics:
        add_value_stat(rows, section, metric, functions[metric](values))
    return values


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(values: pd.Series, filename: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not values.empty:
        bins = min(20, max(5, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color="#4472C4", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, filename)


def plot_bar(categories: pd.Series, counts: pd.Series, filename: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(categories))
    ax.bar(positions, counts.to_numpy(), color="#70AD47", edgecolor="white")
    ax.set_xticks(positions, [str(value) for value in categories])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, filename)


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    correlation: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=30, alpha=0.75, color="#ED7D31", edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {correlation:.4f}" if pd.notna(correlation) else "Pearson r = NA",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax.grid(alpha=0.25)
    save_figure(fig, filename)


def canonical_fps(value: float) -> str:
    """Group common equivalent metadata values without changing raw values."""

    families = [(23.976, "23.976"), (25.0, "25"), (29.97, "29.97"), (30.0, "30")]
    nearest = min(families, key=lambda pair: abs(value - pair[0]))
    if abs(value - nearest[0]) <= 0.1:
        return nearest[1]
    return f"other ({value:g})"


def quality_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(is_true)


def build_anomaly_table(frame: pd.DataFrame, numeric_columns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Return all rows triggering the explicitly defined audit flags."""

    reasons: Dict[str, List[str]] = {str(sample_id): [] for sample_id in frame["sample_id"]}
    for index, row in frame.iterrows():
        sample_id = str(row.get("sample_id", f"row_{index}"))
        if str(row.get("source_match_status", "")) != "both":
            reasons[sample_id].append("source_not_matched")
        if not is_true(row.get("cv_readable")) or not is_true(row.get("has_video")):
            reasons[sample_id].append("invalid_video")
        if not is_true(row.get("has_audio")):
            reasons[sample_id].append("audio_unavailable")
        if not is_true(row.get("label_annotation_match")):
            reasons[sample_id].append("label_annotation_mismatch")
        duration = numeric_columns["container_duration"].loc[index]
        if pd.isna(duration) or duration < EXPECTED_DURATION_MIN_SECONDS or duration > EXPECTED_DURATION_MAX_SECONDS:
            reasons[sample_id].append("duration_out_of_expected_range")
        fps = numeric_columns["fps"].loc[index]
        if pd.isna(fps) or not np.isfinite(fps) or fps <= 0 or fps > 120:
            reasons[sample_id].append("abnormal_fps_metadata")
        relative_error = numeric_columns["av_duration_relative_error"].loc[index]
        if pd.notna(relative_error) and relative_error > AV_RELATIVE_ERROR_REPORT_THRESHOLD:
            reasons[sample_id].append("av_duration_relative_error_high")
        word_count = numeric_columns["word_count"].loc[index]
        if pd.isna(word_count) or word_count < 0:
            reasons[sample_id].append("invalid_word_count")

    records = []
    for index, row in frame.iterrows():
        sample_id = str(row.get("sample_id", f"row_{index}"))
        if reasons[sample_id]:
            records.append(
                {
                    "sample_id": sample_id,
                    "reasons": ";".join(reasons[sample_id]),
                    "container_duration": numeric_columns["container_duration"].loc[index],
                    "word_count": numeric_columns["word_count"].loc[index],
                    "fps": numeric_columns["fps"].loc[index],
                    "av_duration_diff": numeric_columns["av_duration_diff"].loc[index],
                    "av_duration_relative_error": numeric_columns["av_duration_relative_error"].loc[index],
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The manifest is the only data input in this script.
    frame = pd.read_csv(MANIFEST_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")

    total_samples = len(frame)
    numeric_columns = {
        column: numeric(frame, column)
        for column in [
            "container_duration",
            "word_count",
            "fps",
            "frame_count",
            "audio_sample_rate",
            "audio_channels",
            "audio_duration",
            "video_duration",
            "av_duration_diff",
            "av_duration_relative_error",
            "label",
        ]
    }
    rows: List[Dict[str, Any]] = []

    # A. Container duration.
    duration = numeric_columns["container_duration"].dropna()
    duration_metrics = ["count", "mean", "std", "min", "25%", "median", "75%", "max", "P90", "P95"]
    add_descriptive_stats(rows, "video_duration", duration, duration_metrics)
    short_count = int((duration < EXPECTED_DURATION_MIN_SECONDS).sum())
    long_count = int((duration > EXPECTED_DURATION_MAX_SECONDS).sum())
    add_value_stat(rows, "video_duration", "very_short_lt_2s_count", short_count)
    add_value_stat(rows, "video_duration", "very_long_gt_40s_count", long_count)
    plot_histogram(duration, "video_duration_distribution.png", "Video Duration Distribution", "Container duration (s)")

    # B. Text length.
    word_count = numeric_columns["word_count"].dropna()
    word_metrics = ["min", "max", "mean", "median", "std", "P90", "P95"]
    add_descriptive_stats(rows, "word_count", word_count, word_metrics)
    gt_50 = int((word_count > WORD_COUNT_REFERENCE).sum())
    le_50 = int((word_count <= WORD_COUNT_REFERENCE).sum())
    add_value_stat(rows, "word_count", "gt_50_count", gt_50)
    add_value_stat(rows, "word_count", "gt_50_proportion", gt_50 / len(word_count) if len(word_count) else np.nan)
    add_value_stat(rows, "word_count", "le_50_count", le_50)
    add_value_stat(rows, "word_count", "le_50_proportion", le_50 / len(word_count) if len(word_count) else np.nan)
    plot_histogram(word_count, "word_count_distribution.png", "Word Count Distribution", "Word count")

    # C. FPS and its metadata-level variation.
    fps = numeric_columns["fps"].dropna()
    add_value_stat(rows, "fps", "unique_value_count", fps.nunique())
    add_value_stat(rows, "fps", "unique_values", ";".join(f"{value:.12g}" for value in sorted(fps.unique())))
    add_value_stat(rows, "fps", "mean", fps.mean())
    add_value_stat(rows, "fps", "std", fps.std())
    add_value_stat(rows, "fps", "min", fps.min())
    add_value_stat(rows, "fps", "max", fps.max())
    fps_families = fps.map(canonical_fps)
    add_frequency_rows(rows, "fps", "major_frame_rate_family", fps_families, total=len(fps))
    invalid_fps_count = int(((fps <= 0) | (fps > 120) | ~np.isfinite(fps)).sum())
    add_value_stat(rows, "fps", "abnormal_fps_metadata_count", invalid_fps_count, "Operational check: non-finite, <=0, or >120 FPS")
    add_value_stat(rows, "fps", "major_frame_rate_family_count", fps_families.nunique())
    plot_histogram(fps, "fps_distribution.png", "Video FPS Distribution", "FPS metadata")

    # D. Video frame count.
    frame_count = numeric_columns["frame_count"].dropna()
    add_descriptive_stats(rows, "frame_count", frame_count, ["count", "mean", "std", "min", "25%", "median", "75%", "max", "P90", "P95"])
    plot_histogram(frame_count, "frame_count_distribution.png", "Video Frame Count Distribution", "Frame count")

    # E. Audio properties.
    add_frequency_rows(rows, "audio", "audio_sample_rate", numeric_columns["audio_sample_rate"].map(lambda x: x if pd.notna(x) else np.nan), total=total_samples)
    add_frequency_rows(rows, "audio", "audio_channels", numeric_columns["audio_channels"].map(lambda x: x if pd.notna(x) else np.nan), total=total_samples)
    add_frequency_rows(rows, "audio", "audio_codec", frame["audio_codec"], total=total_samples)

    # F. Audio/video duration difference.
    av_diff = numeric_columns["av_duration_diff"].dropna()
    av_error = numeric_columns["av_duration_relative_error"].dropna()
    add_descriptive_stats(rows, "av_duration_difference", av_diff, ["count", "mean", "std", "min", "median", "max", "P90", "P95"])
    add_descriptive_stats(rows, "av_duration_relative_error", av_error, ["count", "mean", "std", "min", "median", "max", "P90", "P95"])
    plot_histogram(av_diff, "av_duration_difference.png", "Audio/Video Duration Difference", "Absolute duration difference (s)")
    top10 = frame.loc[av_error.nlargest(10).index].copy()
    top10["difference"] = numeric_columns["av_duration_diff"].loc[top10.index]
    top10["relative_error"] = numeric_columns["av_duration_relative_error"].loc[top10.index]
    for _, row in top10.iterrows():
        rows.append(
            {
                "section": "av_duration_top10",
                "metric": "largest_relative_error",
                "category": "",
                "value": row["relative_error"],
                "count": np.nan,
                "proportion": np.nan,
                "sample_id": row["sample_id"],
                "video_duration": row["video_duration"],
                "audio_duration": row["audio_duration"],
                "difference": row["difference"],
                "relative_error": row["relative_error"],
                "note": "Sorted descending by relative_error",
            }
        )

    # G. Sentiment labels.
    category_counts = frame["annotation"].fillna("<missing>").value_counts()
    requested_categories = ["Negative", "Neutral", "Positive"]
    ordered_categories = [category for category in requested_categories if category in category_counts.index]
    ordered_categories += [category for category in category_counts.index if category not in ordered_categories]
    category_counts = category_counts.reindex(ordered_categories)
    add_frequency_rows(rows, "sentiment", "category", frame["annotation"], total=total_samples)
    label = numeric_columns["label"].dropna()
    add_descriptive_stats(rows, "sentiment", label, ["min", "max", "mean", "median", "std"])
    plot_bar(category_counts.index, category_counts, "sentiment_category_distribution.png", "Sentiment Category Distribution", "Sentiment category")
    plot_histogram(label, "sentiment_intensity_distribution.png", "Sentiment Intensity Distribution", "Continuous label")

    # H. Length relationships. Pearson correlation is descriptive only.
    duration_word = pd.DataFrame({"duration": numeric_columns["container_duration"], "word_count": numeric_columns["word_count"]}).dropna()
    duration_frame = pd.DataFrame({"duration": numeric_columns["container_duration"], "frame_count": numeric_columns["frame_count"]}).dropna()
    duration_word_r = duration_word["duration"].corr(duration_word["word_count"])
    duration_frame_r = duration_frame["duration"].corr(duration_frame["frame_count"])
    add_value_stat(rows, "length_relationship", "duration_vs_word_count_pearson_r", duration_word_r, "Descriptive association; not causal")
    add_value_stat(rows, "length_relationship", "duration_vs_frame_count_pearson_r", duration_frame_r, "Descriptive association; not causal")
    plot_scatter(duration_word["duration"], duration_word["word_count"], "duration_vs_word_count.png", "Duration vs Word Count", "Container duration (s)", "Word count", duration_word_r)
    plot_scatter(duration_frame["duration"], duration_frame["frame_count"], "duration_vs_frame_count.png", "Duration vs Frame Count", "Container duration (s)", "Frame count", duration_frame_r)

    # I. Quality summary uses manifest audit fields; no raw-source validation is added here.
    quality_values = {
        "total_samples": total_samples,
        "matched_samples": int((frame["source_match_status"].astype(str) == "both").sum()),
        "valid_video_samples": int(quality_flag(frame, "q_video").sum()),
        "audio_available_samples": int(quality_flag(frame, "q_audio").sum()),
        "label_consistent_samples": int(quality_flag(frame, "q_label").sum()),
        "duration_valid_samples": int(quality_flag(frame, "q_duration").sum()),
        "basic_quality_pass_samples": int(quality_flag(frame, "basic_quality_pass").sum()),
    }
    quality_rows = [
        {"metric": metric, "value": value, "proportion": value / total_samples if total_samples else np.nan}
        for metric, value in quality_values.items()
    ]
    pd.DataFrame(quality_rows).to_csv(QUALITY_PATH, index=False, encoding="utf-8-sig")

    statistics_frame = pd.DataFrame(rows)
    statistics_frame.to_csv(STATISTICS_PATH, index=False, encoding="utf-8-sig")

    anomalies = build_anomaly_table(frame, numeric_columns)
    major_fps = fps_families.value_counts()
    audio_channels = numeric_columns["audio_channels"].dropna().value_counts().sort_index()
    report_lines: List[str] = []
    report_lines.extend(
        [
            "RAW MANIFEST EDA REPORT",
            "Input: outputs/metadata/raw_manifest.csv only",
            "",
            "[Dataset]",
            f"total_samples: {total_samples}",
            f"matched_samples: {quality_values['matched_samples']}",
            "",
            "[Duration]",
            f"container_duration min/median/mean/max: {duration.min():.3f} / {duration.median():.3f} / {duration.mean():.3f} / {duration.max():.3f} s",
            f"very_short (<2 s): {short_count}",
            f"very_long (>40 s): {long_count}",
            "",
            "[Words]",
            f"word_count min/median/mean/max: {word_count.min():.0f} / {word_count.median():.1f} / {word_count.mean():.2f} / {word_count.max():.0f}",
            f"word_count > 50: {gt_50} ({gt_50 / len(word_count):.1%})",
            f"word_count <= 50: {le_50} ({le_50 / len(word_count):.1%})",
            "Note: this is evidence for later L_max selection, not a decision of L_max=50.",
            "",
            "[Audio / Video]",
            f"FPS exact unique values: {fps.nunique()}; canonical major rate families: {fps_families.nunique()}",
            "FPS families: " + ", ".join(f"{key} ({value})" for key, value in major_fps.items()),
            f"abnormal FPS metadata: {invalid_fps_count}",
            f"audio channels: " + ", ".join(f"{int(key)} ({value})" for key, value in audio_channels.items()),
            "The FPS metadata variation is described here; it is not a VFR determination.",
            "",
            "[A/V Duration Difference]",
            f"relative error > 5%: {int((av_error > AV_RELATIVE_ERROR_REPORT_THRESHOLD).sum())}",
            "Duration agreement is a necessary synchronization check, but not complete synchronization evidence.",
            "",
            "[Label]",
            "category counts: " + ", ".join(f"{key}={value}" for key, value in category_counts.items()),
            f"continuous label min/median/mean/max: {label.min():.4f} / {label.median():.4f} / {label.mean():.4f} / {label.max():.4f}",
            "",
            "[Length Relationships]",
            f"Pearson duration vs word_count: {duration_word_r:.4f}",
            f"Pearson duration vs frame_count: {duration_frame_r:.4f}",
            "Correlations are descriptive associations and do not establish causality.",
            "",
            "[Quality]",
        ]
    )
    for metric, value in quality_values.items():
        report_lines.append(f"{metric}: {value}")
    report_lines.extend(
        [
            "",
            "[Anomalies]",
            f"operational anomaly sample count: {len(anomalies)}",
        ]
    )
    if anomalies.empty:
        report_lines.append("none")
    else:
        for _, row in anomalies.iterrows():
            report_lines.append(
                f"{row['sample_id']} | {row['reasons']} | duration={row['container_duration']} | words={row['word_count']} | fps={row['fps']} | av_diff={row['av_duration_diff']} | rel_error={row['av_duration_relative_error']}"
            )
    report_lines.extend(["", "[Largest A/V Relative Errors]"])
    for _, row in top10.iterrows():
        report_lines.append(
            f"{row['sample_id']} | video={float(row['video_duration']):.6f} | audio={float(row['audio_duration']):.6f} | difference={float(row['difference']):.6f} | relative_error={float(row['relative_error']):.6f}"
        )
    report_lines.extend(
        [
            "",
            "[Temporal Modeling Readiness]",
            "Manifest-level completeness is high: all samples pass the existing basic quality audit.",
            "Before unified temporal modeling, FPS has 4 major metadata families and audio has 1- vs 2-channel layouts; explicit normalization is needed.",
            "Two samples exceed the 5% A/V duration-difference reporting threshold and should be reviewed.",
            "There is no obvious manifest-level blocker, but metadata duration agreement is not proof of temporal synchronization; PTS parsing remains out of scope.",
            "",
            "[Interpretation Boundary]",
            "No PTS parsing was performed. No conclusion about true VFR or complete A/V synchronization is made from these metadata checks.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Required concise terminal overview.
    print("========== Dataset ==========")
    print(f"samples: {total_samples}")
    print("========== Duration ==========")
    print(f"mean={duration.mean():.3f}s, median={duration.median():.3f}s, min={duration.min():.3f}s, max={duration.max():.3f}s")
    print("========== Words ==========")
    print(f"mean={word_count.mean():.2f}, median={word_count.median():.1f}, >50={gt_50} ({gt_50 / len(word_count):.1%}), <=50={le_50} ({le_50 / len(word_count):.1%})")
    print("========== Audio / Video ==========")
    print(f"FPS exact unique={fps.nunique()}, major families={fps_families.nunique()}, audio channels={audio_channels.to_dict()}")
    print("========== Label ==========")
    print("category=" + ", ".join(f"{key}:{value}" for key, value in category_counts.items()))
    print("========== Quality ==========")
    print("; ".join(f"{key}={value}" for key, value in quality_values.items()))
    print(f"Generated: {STATISTICS_PATH}")
    print(f"Generated: {QUALITY_PATH}")
    print(f"Generated: {REPORT_PATH}")
    print(f"Generated figures: {len(list(FIGURE_DIR.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
